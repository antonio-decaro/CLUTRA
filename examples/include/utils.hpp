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

struct Options {
  bool print_output = false;
  bool validate = false;
  bool profiling_detail = false;
  bool binary_format = false;
  bool matrix_market = false;
  bool undirected = false;
  bool local_stealing = false;
  bool global_stealing = false;
  std::optional<int> stealing_chunk_size;
  bool random_source = true;
  std::string path;
  int cluster_size = 4;
  size_t source = 0;
};

struct CLIHandles {
  CLI::Option* source_opt = nullptr;
  CLI::Option* stealing_opt = nullptr;
  CLI::Option* global_stealing_opt = nullptr;
  CLI::Option* stealing_chunk_size_opt = nullptr;
  CLI::Option* cluster_size_opt = nullptr;
};

inline CLIHandles configureBaseCLI(CLI::App& app, Options& opts) {
  CLIHandles handles;
  auto binary_flag = app.add_flag("-b,--binary", opts.binary_format, "Treat input as binary CSR format");
  auto matrix_flag = app.add_flag("-m,--matrix-market", opts.matrix_market, "Treat input as Matrix Market format");
  if (binary_flag && matrix_flag) {
    binary_flag->excludes(matrix_flag);
    matrix_flag->excludes(binary_flag);
  }

  app.add_flag("-p,--print", opts.print_output, "Print algorithm output to stdout");
  app.add_flag("-v,--validate", opts.validate, "Validate algorithm output against CPU implementation");
  app.add_flag("-D,--detail", opts.profiling_detail, "Print detailed profiling info (per-event timings)");
  app.add_flag("-u,--undirected", opts.undirected, "Treat input COO as an undirected graph");
  handles.cluster_size_opt = app.add_option("-c,--cluster-size", opts.cluster_size, "Set the cluster size for intra-cluster work stealing (default: 4)");
  handles.cluster_size_opt->check(CLI::Range(1, 8));
  handles.stealing_opt = app.add_flag("-t,--stealing", opts.local_stealing, "Enable local (intra-cluster) work stealing in the advance operator");
  handles.global_stealing_opt = app.add_flag("-g,--global-stealing", opts.global_stealing, "Enable global stealing (implies local + inter-cluster stealing)");
  handles.stealing_opt->excludes(handles.global_stealing_opt);
  handles.global_stealing_opt->excludes(handles.stealing_opt);
  handles.stealing_chunk_size_opt = app.add_option(
      "--chunk-size",
      opts.stealing_chunk_size,
      "Set the stealing chunk size");
  handles.stealing_chunk_size_opt->check(CLI::PositiveNumber);

  handles.source_opt = app.add_option("-s,--source", opts.source, "Specify the source vertex");
  handles.source_opt->check(CLI::NonNegativeNumber);

  app.add_option("graph", opts.path, "Path to the graph file")->required();

  return handles;
}

inline void finalizeGraphOptions(Options& opts, const CLIHandles& handles) {
  if (handles.source_opt && handles.source_opt->count() > 0) {
    opts.random_source = false;
  } else {
    opts.random_source = true;
  }
  opts.local_stealing =
      (handles.stealing_opt && handles.stealing_opt->count() > 0);
  opts.global_stealing =
      (handles.global_stealing_opt && handles.global_stealing_opt->count() > 0);
}

inline clutra::stealer::StealerConfig getStealingConfig(const Options& opts) {
  clutra::stealer::StealerConfig config{};
  config.intra_cluster_stealing_enabled = opts.local_stealing || opts.global_stealing;
  config.inter_cluster_stealing_enabled = opts.global_stealing;
  if (opts.stealing_chunk_size.has_value()) {
    config.stealing_chunk_size = *opts.stealing_chunk_size;
    config.global_stealing_chunk_size = *opts.stealing_chunk_size;
  }
  config.preferred_cluster_size = opts.cluster_size;
  return config;
}

template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> readCSR(const Options& opts, clutra::graph::Properties* properties = nullptr) {
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

inline void printStealingOptions(const Options& opts, bool header = true, bool footer = true) {
  if (header) {
    std::cerr << "-----------------------------------" << std::endl;
  }
  std::cerr << std::left;
  const bool any_stealing = opts.local_stealing || opts.global_stealing;
  std::cerr << std::setw(26) << "Stealing enabled:" << std::setw(10) << (any_stealing ? "yes" : "no") << std::endl;
  std::cerr << std::setw(26) << "Global stealing:" << std::setw(10) << (opts.global_stealing ? "yes" : "no") << std::endl;
  std::cerr << std::setw(26) << "Stealing chunk size:" << std::setw(10)
            << (opts.stealing_chunk_size.has_value() ? std::to_string(*opts.stealing_chunk_size) : "default")
            << std::endl;
  std::cerr << std::setw(26) << "Cluster size:" << std::setw(10) << opts.cluster_size << std::endl;
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

bool checkOrderedCSR(
    const clutra::formats::CSR<float, uint32_t, uint32_t> &csr) {
  const auto &row_offsets = csr.getRowOffsets();
  const auto &col_indices = csr.getColumnIndices();

  const size_t vertex_count = row_offsets.size() - 1;

  for (size_t u = 0; u < vertex_count; ++u) {
    const auto start = row_offsets[u];
    const auto end = row_offsets[u + 1];
    for (size_t idx = start + 1; idx < end; ++idx) {
      if (col_indices[idx - 1] >= col_indices[idx]) {
        return false;
      }
    }
  }
  return true;
}

void sortCSR(clutra::formats::CSR<float, uint32_t, uint32_t> &csr) {
  const auto &row_offsets = csr.getRowOffsets();
  auto &col_indices = csr.getColumnIndices();
  auto &values = csr.getValues();

  const size_t vertex_count = row_offsets.size() - 1;

  for (size_t u = 0; u < vertex_count; ++u) {
    const auto start = row_offsets[u];
    const auto end = row_offsets[u + 1];

    // Create a vector of pairs (col_index, value)
    std::vector<std::pair<uint32_t, float>> neighbors;
    for (size_t idx = start; idx < end; ++idx) {
      neighbors.emplace_back(col_indices[idx], values[idx]);
    }

    // Sort the neighbors based on col_index
    std::sort(
        neighbors.begin(), neighbors.end(),
        [](const std::pair<uint32_t, float> &a,
           const std::pair<uint32_t, float> &b) { return a.first < b.first; });

    // Write back the sorted neighbors
    for (size_t idx = start; idx < end; ++idx) {
      col_indices[idx] = neighbors[idx - start].first;
      values[idx] = neighbors[idx - start].second;
    }
  }
}
