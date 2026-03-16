#include "utils.hpp"
#include <clutra.hpp>
#include <iostream>

constexpr size_t MAX_THREADS = 1 << 20;

template <typename GraphT>
bool validate(const GraphT& graph, const int* device_distances, const uint source) {
  std::vector<uint32_t> distances(graph.getVertexCount(), graph.getVertexCount() + 1);
  std::vector<uint32_t> in_frontier;
  std::vector<uint32_t> out_frontier;
  in_frontier.push_back(source);
  distances[source] = 0;

  auto* row_offsets = graph.getRowOffsets();
  auto* col_indices = graph.getColumnIndices();

  size_t iter = 0;
  size_t mismatches = 0;
  while (in_frontier.size()) {
    for (size_t i = 0; i < in_frontier.size(); i++) {
      auto vertex = in_frontier[i];

      auto start = row_offsets[vertex];
      auto end = row_offsets[vertex + 1];

      for (size_t j = start; j < end; j++) {
        auto neighbor = col_indices[j];
        if (distances[neighbor] == graph.getVertexCount() + 1) {
          distances[neighbor] = distances[vertex] + 1;
          if (distances[neighbor] != device_distances[neighbor]) {
            mismatches++;
          }
          out_frontier.push_back(neighbor);
        }
      }
    }
    std::swap(in_frontier, out_frontier);
    out_frontier.clear();
    iter++;
  }
  if (mismatches) {
    std::cout << "Mismatches: " << mismatches << std::endl;
  }
  return mismatches == 0;
}

int main(int argc, char** argv) {

  Options opts;
  CLI::App app{"CLUTRA BFS"};
  auto cli_handles = configureBaseCLI(app, opts);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, cli_handles);

  std::cerr << "[*] Reading CSR" << std::endl;
  clutra::graph::Properties properties;
  auto csr = readCSR<float, uint32_t, uint32_t>(opts, &properties);
  std::cerr << "[*] CSR Building Graph" << std::endl;
  auto graph = clutra::graph::createGraph(csr, properties);
  printGraphInfo(graph);
  auto stealer_config = getStealingConfig(opts);
  printStealingOptions(stealer_config, false);

  clutra::frontier::FrontierMLB<uint32_t> in_frontier(graph.getVertexCount());
  clutra::frontier::FrontierMLB<uint32_t> out_frontier(graph.getVertexCount());

  if (opts.random_source) {
    opts.source = getRandomSource(graph.getVertexCount());
  }

  int* distances;
  CUDA_CHECK(cudaMallocManaged(&distances, graph.getVertexCount() * sizeof(int)));
  CUDA_CHECK(cudaMemset(distances, -1, graph.getVertexCount() * sizeof(int)));

  size_t* work;
  if (opts.measure_work) {
    CUDA_CHECK(cudaMallocManaged(&work, MAX_THREADS * sizeof(size_t)));
    CUDA_CHECK(cudaMemset(work, 0, MAX_THREADS * sizeof(size_t)));
  }

  distances[opts.source] = 0;
  in_frontier.insert(opts.source);

  int iter = 0;

  clutra::stealer::BasicStealer stealer(stealer_config);

  std::cout << "[*] Running BFS from source vertex " << opts.source << std::endl;
  while (!in_frontier.empty()) {
    // std::cout << "[*] BFS Iteration " << iter << ", Frontier Size: " << in_frontier.getOutDegree(graph) << std::endl;
    clutra::operators::advance::push(
        graph, in_frontier, out_frontier, stealer, clutra::operators::advance::load_balance::block_mapped,
        [iter, distances, work, measure_work = opts.measure_work] __device__(auto u, auto v, auto e, auto w) {
          if (measure_work) {
            work[blockIdx.x * blockDim.x + threadIdx.x]++;
          }
          if (distances[v] == -1) {
            distances[v] = iter + 1;
            return true;
          }
          return false;
        });

    clutra::frontier::FrontierMLB<uint32_t>::swap(in_frontier, out_frontier);
    out_frontier.clear();
    iter++;
  }
  std::cout << "[*] BFS complete in " << iter << " iterations." << std::endl;

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
    std::cout << "Validation Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count()
              << " ms" << std::endl;
  }

  cudaFree(distances);

  if (opts.measure_work) {
    std::vector<size_t> h_work_per_thread(MAX_THREADS);
    CUDA_CHECK(cudaMemcpy(h_work_per_thread.data(), work, sizeof(size_t) * MAX_THREADS, cudaMemcpyDeviceToHost));
    cudaFree(work);
    printArray(h_work_per_thread, "Work per thread: ");
  }

  clutra::profile::KernelProfilerManager::instance().printSummary(opts.profiling_detail);
}
