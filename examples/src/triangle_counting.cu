#include "utils.hpp"
#include <clutra.hpp>
#include <cstdint>
#include <iostream>
#include <thrust/device_vector.h>

template <typename DeviceGraphT> struct MergePathsFunctor {
  DeviceGraphT graph_dev;
  int *edges;

  template <typename U, typename V, typename E, typename W>
  __device__ bool operator()(U u, V v, E e, W w) const {
    auto src_it = graph_dev.begin(u);
    auto src_end = graph_dev.end(u);
    auto dst_it = graph_dev.begin(v);
    auto dst_end = graph_dev.end(v);
    while (src_it != src_end && dst_it != dst_end) {
      if (*src_it == *dst_it) {
        // Found a common neighbor
        atomicAdd(&edges[u], 1);
        ++src_it;
        ++dst_it;
      } else if (*src_it < *dst_it) {
        ++src_it;
      } else {
        ++dst_it;
      }
    }
    return false;
  }
};

template <typename DeviceGraphT> struct BinarySearchFunctor {
  DeviceGraphT graph_dev;
  int *edges;

  template <typename U, typename V, typename E, typename W>
  __device__ bool operator()(U u, V v, E e, W w) const {
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
    for (int i = 0; i < a_deg; ++i) {
      auto x = col_indices[a_start + i];
      if (b_deg == 0) {
        continue;
      }
      if (b_deg == 1) {
        if (x == col_indices[b_start]) {
          atomicAdd(&edges[u], 1);
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
        } else { // x == y
          edges[threadIdx.x + (blockIdx.x * blockDim.x)] += 1;
          found = true;
          break;
        }
      }

      if (!found) {
        if (x == col_indices[b_start + bottom] ||
            x == col_indices[b_start + top]) {
          edges[threadIdx.x + (blockIdx.x * blockDim.x)] += 1;
        }
      }
    }
    return false;
  }
};

template <typename GraphT> size_t validate(const GraphT &graph) {
  const auto *row_offsets = graph.getRowOffsets();
  const auto *col_indices = graph.getColumnIndices();

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

int main(int argc, char **argv) {
  Options opts;
  CLI::App app{"CLUTRA Triangle Counting (TC)"};
  auto cli_handles = configureBaseCLI(app, opts);
  std::string tc_method = "merge";
  app.add_option("--method", tc_method,
                 "Triangle counting method: merge or binary (default: merge)")
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
  printStealingOptions(opts, false);

  auto stealer_config = getStealingConfig(opts);
  clutra::stealer::BasicStealer stealer(stealer_config);

  auto graph_dev = graph.getDeviceGraph();

  const size_t thread_count = 264 * 512 * 32;
  int *edges;
  cudaMalloc(&edges, sizeof(int) * graph.getVertexCount());
  cudaMemset(edges, 0, sizeof(int) * graph.getVertexCount());

  std::cout << "[*] Running TC with method: " << tc_method << std::endl;
  if (tc_method == "binary") {
    clutra::operators::advance::graph(graph, stealer,
                                      BinarySearchFunctor{graph_dev, edges});
  } else {
    clutra::operators::advance::graph(graph, stealer,
                                      MergePathsFunctor{graph_dev, edges});
  }

  clutra::profile::KernelProfiler profiler("reduce_triangles");
  std::uint64_t device_triangles = thrust::reduce(
      thrust::device_ptr<int>(edges),
      thrust::device_ptr<int>(edges + graph.getVertexCount()), 0LL);
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
      std::cout << " (CPU: " << validation_result
                << ", GPU: " << device_triangles << ")";
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     validation_end - validation_start)
                     .count()
              << " ms" << std::endl;
  }

  cudaFree(edges);

  clutra::profile::KernelProfilerManager::instance().printSummary(
      opts.profiling_detail);
}
