#include "utils.hpp"
#include <clutra.hpp>
#include <cstdint>
#include <iostream>
#include <thrust/device_vector.h>

template <typename DeviceGraphT>
struct MergePathsFunctor {
  DeviceGraphT graph_dev;
  bool directed;
  int* edges;

  template <typename U, typename V, typename E, typename W>
  __device__ bool operator()(U u, V v, E e, W w) const {
    (void)e;
    (void)w;
    if (!directed && u >= v) {
      return false;  // Process each edge only once
    }
    auto src_it = graph_dev.begin(u);
    auto src_end = graph_dev.end(u);
    auto dst_it = graph_dev.begin(v);
    auto dst_end = graph_dev.end(v);
    int local_triangles = 0;
    while (src_it != src_end && dst_it != dst_end) {
      if (*src_it == *dst_it) {
        ++local_triangles;
        ++src_it;
        ++dst_it;
      } else if (*src_it < *dst_it) {
        ++src_it;
      } else {
        ++dst_it;
      }
    }
    if (local_triangles > 0) {
      atomicAdd(&edges[u], local_triangles);
    }
    return false;
  }
};

template <typename DeviceGraphT>
struct BinarySearchFunctor {
  DeviceGraphT graph_dev;
  bool directed;
  int* edges;

  template <typename U, typename V, typename E, typename W>
  __device__ bool operator()(U u, V v, E e, W w) const {
    (void)e;
    (void)w;
    if (!directed && u >= v) {
      return false;  // Process each edge only once
    }
    auto src_start = graph_dev.getFirstNeighbor(u);
    auto dst_start = graph_dev.getFirstNeighbor(v);
    auto src_deg = graph_dev.getDegree(u);
    auto dst_deg = graph_dev.getDegree(v);

    // Ensure |A| <= |B|
    auto a_start = src_start;
    auto a_deg = src_deg;
    auto b_start = dst_start;
    auto b_deg = dst_deg;
    if (src_deg > dst_deg) {
      a_start = dst_start;
      a_deg = dst_deg;
      b_start = src_start;
      b_deg = src_deg;
    }

    auto col_indices = graph_dev.getColumnIndices();
    int local_triangles = 0;
    for (int i = 0; i < a_deg; ++i) {
      auto x = col_indices[a_start + i];
      if (b_deg == 0) {
        continue;
      }
      if (b_deg == 1) {
        if (x == col_indices[b_start]) {
          ++local_triangles;
        }
        continue;
      }

      int bottom = 0;
      int top = b_deg - 1;
      bool found = false;
      while (bottom + 1 < top) {
        int mid = (top + bottom) >> 1;
        auto y = col_indices[b_start + mid];
        if (x < y) {
          top = mid;
        } else if (x > y) {
          bottom = mid;
        } else {  // x == y
          ++local_triangles;
          found = true;
          break;
        }
      }

      if (!found) {
        if (x == col_indices[b_start + bottom] || x == col_indices[b_start + top]) {
          ++local_triangles;
        }
      }
    }
    if (local_triangles > 0) {
      atomicAdd(&edges[u], local_triangles);
    }
    return false;
  }
};

template <typename GraphT>
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
      if (u >= v)
        continue;
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

  return cpu_triangles;  // Each triangle is counted three times
}

int main(int argc, char** argv) {
  Options opts;
  CLI::App app{"CLUTRA Triangle Counting (TC)"};
  auto cli_handles = configureBaseCLI(app, opts);
  std::string tc_method = "merge";
  app.add_option("--method", tc_method, "Triangle counting method: merge or binary (default: merge)")
      ->check(CLI::IsMember({"merge", "binary"}));
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, cli_handles);

  std::cerr << "[*] Reading CSR" << std::endl;
  clutra::graph::Properties properties;
  auto csr = readCSR<float, uint32_t, uint32_t>(opts, &properties);
  std::cerr << "[*] Checking CSR is ordered" << std::endl;
  if (!checkOrderedCSR(csr)) {
    std::cerr << "[*] CSR not ordered. Sorting..." << std::endl;
    sortCSR(csr);
  }
  std::cerr << "[*] CSR sorted" << std::endl;
  std::cerr << "[*] CSR Building Graph" << std::endl;

  auto graph = clutra::graph::createGraph(csr, properties);
  printGraphInfo(graph);

  auto stealer_config = getStealingConfig(opts);
  printStealingOptions(stealer_config, false);
  clutra::stealer::BasicStealer stealer(stealer_config);

  auto graph_dev = graph.getDeviceGraph();

  int* edges;
  cudaMalloc(&edges, sizeof(int) * graph.getVertexCount());
  cudaMemset(edges, 0, sizeof(int) * graph.getVertexCount());

  bool directed = graph.getProperties().directed;
  std::cout << "[*] Running TC with method: " << tc_method << std::endl;
  if (tc_method == "binary") {
    clutra::operators::advance::graph(graph, stealer, clutra::operators::advance::load_balance::block_mapped,
                                      BinarySearchFunctor{graph_dev, directed, edges});
  } else {
    clutra::operators::advance::graph(graph, stealer, clutra::operators::advance::load_balance::block_mapped,
                                      MergePathsFunctor{graph_dev, directed, edges});
  }

  clutra::profile::KernelProfiler profiler("reduce_triangles");
  std::uint64_t device_triangles =
      thrust::reduce(thrust::device_ptr<int>(edges), thrust::device_ptr<int>(edges + graph.getVertexCount()), 0LL);
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
    std::cout << "Validation Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count()
              << " ms" << std::endl;
  }

  cudaFree(edges);

  clutra::profile::KernelProfilerManager::instance().printSummary(opts.profiling_detail);
}
