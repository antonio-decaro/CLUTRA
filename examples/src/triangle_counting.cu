#include <clutra.hpp>
#include "utils.hpp"
#include <cstdint>
#include <thrust/device_vector.h>
#include <iostream>

// Prevents the compiler from optimizing away synthetic work in the BFS kernel.
__device__ unsigned long long g_work_sink = 0;

template<typename GraphT>
size_t validate(const GraphT& graph) {
  const auto* row_offsets = graph.getRowOffsets();
  const auto* col_indices = graph.getColumnIndices();

  const size_t vertex_count = graph.getVertexCount();
  std::uint64_t cpu_triangles = 0;

  for (size_t u = 0; u < vertex_count; ++u) {
    const auto u_start = row_offsets[u];
    const auto u_end = row_offsets[u + 1];
    for (size_t e = u_start; e < u_end; ++e) {
      const auto v = col_indices[e];
      const auto v_start = row_offsets[v];
      const auto v_end = row_offsets[v + 1];
      auto u_it = u_start;
      auto v_it = v_start;
      while (u_it < u_end && v_it < v_end) {
        const auto u_neighbor = col_indices[u_it];
        const auto v_neighbor = col_indices[v_it];
        if (u_neighbor == v_neighbor) {
          ++cpu_triangles;
          ++u_it;
          ++v_it;
        } else if (u_neighbor < v_neighbor) {
          ++u_it;
        } else {
          ++v_it;
        }
      }
    }
  }

  return cpu_triangles / 3; // Each triangle is counted three times
}

int main(int argc, char** argv) {
  Options opts;
  CLI::App app{"CLUTRA Triangle Counting (TC)"};
  auto cli_handles = configureBaseCLI(app, opts);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, cli_handles);

  std::cerr << "[*] Reading CSR" << std::endl;
  clutra::graph::Properties properties;
  auto csr = readCSR<float, uint32_t, uint32_t>(opts, &properties);
  std::cerr << "[*] CSR Building Graph" << std::endl;
  auto graph = clutra::graph::createGraph(csr, properties);
  printGraphInfo(graph);
  printStealingOptions(opts, false);
  
  auto stealer_config = getStealingConfig(opts);
  clutra::stealer::BasicStealer stealer(stealer_config);

  auto graph_dev = graph.getDeviceGraph();

  const size_t thread_count = 264 * 512 * 32;
  int* edges;
  cudaMalloc(&edges, sizeof(int) * thread_count);
  cudaMemset(edges, 0, sizeof(int) * thread_count);
  
  std::cout << "[*] Running TC" << std::endl;
  clutra::operators::advance::graph(graph, stealer, 
    [=] __device__ (auto u, auto v, auto e, auto w) {
      // for (int i = 0; i < 1000; i++) {
        auto src_it = graph_dev.begin(u);
        auto src_end = graph_dev.end(u);
        auto dst_it = graph_dev.begin(v);
        auto dst_end = graph_dev.end(v);
        while (src_it != src_end && dst_it != dst_end) {
          if (*src_it == *dst_it) {
            // Found a common neighbor
            edges[threadIdx.x + blockIdx.x * blockDim.x] += 1;
            ++src_it;
            ++dst_it;
          } else if (*src_it < *dst_it) {
            ++src_it;
          } else {
            ++dst_it;
          }
        }
      // }
      return false;
  });
  
  clutra::profile::KernelProfiler profiler("reduce_triangles");
  std::uint64_t device_triangles = thrust::reduce(thrust::device_ptr<int>(edges), thrust::device_ptr<int>(edges + thread_count), 0LL);
  device_triangles /= 3; // Each triangle is counted three times
  profiler.stop();
  
  std::cout << "[*] TC completed" << std::endl;
  std::cout << "[*] Triangles: " << device_triangles << std::endl;


  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    auto validation_result = validate(graph);
    if (validation_result != device_triangles) {
      std::cout << failString();
      std::cout << " (CPU: " << validation_result << ", GPU: " << device_triangles << ")";
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  // for (int i = 0; i < thread_count; i++) {
  //   if (edges[i] > 0) {
  //     std::cout << "[" << i << "]" << edges[i] << " ";
  //   }
  // }
  // std::cout << std::endl;
  
  cudaFree(edges);
  
  clutra::profile::KernelProfilerManager::instance().printSummary(opts.profiling_detail);
}
