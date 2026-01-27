/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <CLI/CLI.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

#include <clutra.hpp>

struct GraphOptions {
  bool print_output = false;
  bool validate = false;
  bool binary_format = false;
  bool matrix_market = false;
  bool undirected = false;
  bool stealing = false;
  std::optional<int> stealing_chunk_size;
  bool random_source = true;
  std::string path;
  size_t source = 0;
};

struct CLIHandles {
  CLI::Option* source_opt = nullptr;
  CLI::Option* stealing_opt = nullptr;
};

inline CLIHandles configureBaseCLI(CLI::App& app, GraphOptions& opts) {
  CLIHandles handles;
  auto binary_flag = app.add_flag("-b,--binary", opts.binary_format, "Treat input as binary CSR format");
  auto matrix_flag = app.add_flag("-m,--matrix-market", opts.matrix_market, "Treat input as Matrix Market format");
  if (binary_flag && matrix_flag) {
    binary_flag->excludes(matrix_flag);
    matrix_flag->excludes(binary_flag);
  }

  app.add_flag("-p,--print", opts.print_output, "Print algorithm output to stdout");
  app.add_flag("-v,--validate", opts.validate, "Validate algorithm output against CPU implementation");
  app.add_flag("-u,--undirected", opts.undirected, "Treat input COO as an undirected graph");
  handles.stealing_opt = app.add_option(
      "-t,--stealing",
      opts.stealing_chunk_size,
      "Enable work stealing in the advance operator (optional chunk size)");
  handles.stealing_opt->expected(0, 1);
  handles.stealing_opt->check(CLI::PositiveNumber);

  handles.source_opt = app.add_option("-s,--source", opts.source, "Specify the source vertex");
  handles.source_opt->check(CLI::NonNegativeNumber);

  app.add_option("graph", opts.path, "Path to the graph file")->required();

  return handles;
}

inline void finalizeGraphOptions(GraphOptions& opts, const CLIHandles& handles) {
  if (handles.source_opt && handles.source_opt->count() > 0) {
    opts.random_source = false;
  } else {
    opts.random_source = true;
  }
  if (handles.stealing_opt && handles.stealing_opt->count() > 0) {
    opts.stealing = true;
  } else {
    opts.stealing = false;
  }
}

template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> readCSR(const GraphOptions& opts, clutra::graph::Properties* properties = nullptr) {
  clutra::formats::CSR<ValueT, IndexT, OffsetT> csr;
  clutra::graph::Properties local_properties;
  auto* props = properties ? properties : &local_properties;
  if (opts.binary_format) {
    std::ifstream file(opts.path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << opts.path << std::endl;
      exit(1);
    }
    csr = clutra::io::csr::fromBinary<ValueT, IndexT, OffsetT>(file, props);
  } else if (opts.matrix_market) {
    csr = clutra::io::csr::fromMM<ValueT, IndexT, OffsetT>(opts.path, props);
  } else {
    throw std::runtime_error("Input format not specified or unsupported");
  }

  return csr;
}

template<typename T>
void printFrontier(const T& frontier, std::string prefix = "") {
  using bitmap_type = typename T::bitmap_type;
  const auto words = frontier.getBitmapSize();
  if (words == 0) {
    std::cout << prefix << "<empty frontier>" << std::endl;
    return;
  }
  std::vector<bitmap_type> host(words);
  CUDA_CHECK(cudaMemcpy(host.data(), frontier.getDeviceFrontier().getData(), words * sizeof(bitmap_type), cudaMemcpyDeviceToHost));
  const auto range = frontier.getBitmapRange();
  std::cout << prefix;
  for (int bit = static_cast<int>(words * range) - 1; bit >= 0; --bit) {
    const auto word_idx = static_cast<size_t>(bit) / range;
    const auto offset = static_cast<size_t>(bit) % range;
    const auto is_set = (host[word_idx] >> offset) & static_cast<bitmap_type>(1);
    std::cout << (is_set ? "1" : "0");
  }
  std::cout << std::endl;
}

inline size_t getRandomSource(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, size - 1);
  return dis(gen);
}

template<typename GraphT>
void printGraphInfo(const GraphT& g, bool header = true, bool footer = true) {
  if (header) {
    std::cerr << "-----------------------------------" << std::endl;
  }
  std::cerr << std::left;
  std::cerr << std::setw(17) << "Vertex count:" << std::setw(10) << g.getVertexCount() << std::endl;
  std::cerr << std::setw(17) << "Edge count:" << std::setw(10) << g.getEdgeCount() << std::endl;
  std::cerr << std::setw(17) << "Average degree:" << std::setw(10) << g.getEdgeCount() / g.getVertexCount() << std::endl;
  std::cerr << std::setw(17) << "Directed:" << std::setw(10) << (g.getProperties().directed ? "yes" : "no") << std::endl;
  if (footer) {
    std::cerr << "-----------------------------------" << std::endl;
  }
}

inline void printStealingOptions(const GraphOptions& opts, bool header = true, bool footer = true) {
  if (header) {
    std::cerr << "-----------------------------------" << std::endl;
  }
  std::cerr << std::left;
  std::cerr << std::setw(26) << "Stealing enabled:" << std::setw(10) << (opts.stealing ? "yes" : "no") << std::endl;
  std::cerr << std::setw(26) << "Stealing chunk size:" << std::setw(10)
            << (opts.stealing_chunk_size.has_value() ? std::to_string(*opts.stealing_chunk_size) : "default")
            << std::endl;
  if (footer) {
    std::cerr << "-----------------------------------" << std::endl;
  }
}

inline void printDeviceInfo(std::string prefix = "") {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  std::cerr << prefix << "Using device " << device_id << ": " << device_prop.name << " with " << device_prop.totalGlobalMem / (1024 * 1024)
            << " MB of global memory." << std::endl;
}

inline bool isConsoleOutput() { return static_cast<int>(static_cast<int>(isatty(STDOUT_FILENO) != 0)) != 0; }

inline std::string successString() {
  if (!isConsoleOutput()) { return "Success"; }
  return "\033[1;32mSuccess\033[0m";
}

inline std::string failString() {
  if (!isConsoleOutput()) { return "Failed"; }
  return "\033[1;31mFailed\033[0m";
}
