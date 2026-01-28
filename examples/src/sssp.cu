#include <clutra.hpp>
#include "utils.hpp"
#include <queue>
#include <iostream>
#include <limits>

__device__ __forceinline__ float atomicMinFloat(float* addr, float val) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i;
  while (true) {
    float old_f = __int_as_float(old);
    if (old_f <= val) {
      return old_f;
    }
    int assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    if (old == assumed) {
      return __int_as_float(old);
    }
  }
}

template<typename VertexT, typename WeightT>
class Prioritize {
public:
  bool operator()(std::pair<VertexT, WeightT>& p1, std::pair<VertexT, WeightT>& p2) { return p1.second > p2.second; }
};

template<typename GraphT>
bool validate(const GraphT& graph, float* device_distances, uint source) {
  auto* row_offsets = graph.getRowOffsets();
  auto* column_indices = graph.getColumnIndices();
  auto* nonzero_values = graph.getValues();

  std::vector<float> distances(graph.getVertexCount(), std::numeric_limits<float>::infinity());
  distances[source] = 0;

  std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, Prioritize<uint32_t, float>> pq;
  pq.push(std::make_pair(source, 0.0));

  while (!pq.empty()) {
    std::pair<uint32_t, float> curr = pq.top();
    pq.pop();

    uint32_t curr_node = curr.first;
    float curr_dist = curr.second;

    uint32_t start = row_offsets[curr_node];
    uint32_t end = row_offsets[curr_node + 1];

    for (uint32_t offset = start; offset < end; offset++) {
      uint32_t neib = column_indices[offset];
      float new_dist = curr_dist + nonzero_values[offset];
      if (new_dist < distances[neib]) {
        distances[neib] = new_dist;
        pq.push(std::make_pair(neib, new_dist));
      }
    }
  }

  for (auto i = 0; i < graph.getVertexCount(); i++) {
    if (distances[i] != device_distances[i]) {
      std::cerr << "Mismatch at vertex " << i << " | Expected: " << distances[i] << " | Got: " << device_distances[i] << std::endl;
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {

  Options opts;
  CLI::App app{"CLUTRA SSSP"};
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
  
  clutra::frontier::FrontierMLB<uint32_t> in_frontier(graph.getVertexCount());
  clutra::frontier::FrontierMLB<uint32_t> out_frontier(graph.getVertexCount());

  if (opts.random_source) { opts.source = getRandomSource(graph.getVertexCount()); }
  
  float* distances;
  cudaMallocManaged(&distances, graph.getVertexCount() * sizeof(float));
  std::fill(distances, distances + graph.getVertexCount(), std::numeric_limits<float>::infinity());
  
  distances[opts.source] = 0.0f;
  in_frontier.insert(opts.source);
  
  int iter = 0;

  auto stealer_config = getStealingConfig(opts);
  clutra::stealer::BasicStealer stealer(stealer_config);
  
  std::cout << "[*] Running SSSP from source vertex " << opts.source << std::endl;
  while (!in_frontier.empty()) {
    // std::cout << "[*] SSSP Iteration " << iter << ", Frontier Size: " << in_frontier.getOutDegree(graph) << std::endl;
    clutra::operators::advance::push(graph, in_frontier, out_frontier, stealer,
      [iter, distances] __device__ (auto u, auto v, auto e, auto w) {
        float source_distance = distances[u];
        float distance_to_neighbor = source_distance + static_cast<float>(w);

        float recovered_distance = atomicMinFloat(&distances[v], distance_to_neighbor);
        return (distance_to_neighbor < recovered_distance);
      }
    );

    clutra::frontier::FrontierMLB<uint32_t>::swap(in_frontier, out_frontier);
    out_frontier.clear();
    iter++;
  }
  std::cout << "[*] SSSP complete in " << iter << " iterations." << std::endl;

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(graph, distances, opts.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }
  
  cudaFree(distances);

  clutra::profile::KernelProfilerManager::instance().printSummary();
}
