#include "utils.hpp"
#include <clutra.hpp>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cg = cooperative_groups;

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

template <size_t Size>
struct LocalHotRing {
  __device__ bool empty() const { return head == tail; }

  __device__ bool full() const { return ((tail + 1) % Size) == head; }

  __device__ bool push(uint32_t value) {
    if (full())
      return false;
    int pos = atomicAdd(&tail, 1) % Size;
    data[pos] = value;
    return true;
  }

  __device__ bool pop(uint32_t& value) {
    if (empty())
      return false;
    int pos = atomicAdd(&head, 1) % Size;
    value = data[pos];
    return true;
  }

  uint32_t data[Size];
  uint32_t head;
  uint32_t tail;
};

template <size_t Size, size_t BlockSize>
struct CTAHotRings {
  __device__ LocalHotRing<Size>& getHotRing(size_t warp_id) { return rings[warp_id]; }

  LocalHotRing<Size> rings[BlockSize / 32];
};

class GlobalWorkQueue {
public:
  __device__ uint32_t* getData(size_t cluster_id = 0) { return data + (cluster_id * cluster_size); }

  __device__ uint32_t* getHead(size_t cluster_id) { return heads + cluster_id; }

  __device__ uint32_t* getTail(size_t cluster_id) { return tails + cluster_id; }

  __device__ clutra::detail::atomic::TicketLock* getLock(size_t cluster_id) { return locks + cluster_id; }

  __host__ static GlobalWorkQueue allocateWorkQueue(size_t nodes, size_t num_clusters) {
    uint32_t *data, *heads, *tails;
    clutra::detail::atomic::TicketLock* locks;
    CUDA_CHECK(cudaMalloc(&data, nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&heads, num_clusters * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tails, num_clusters * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(heads, 0, num_clusters * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(tails, 0, num_clusters * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&locks, num_clusters * sizeof(clutra::detail::atomic::TicketLock)));

    GlobalWorkQueue wq;
    wq.data = data;
    wq.heads = heads;
    wq.tails = tails;
    wq.locks = locks;
    wq.num_clusters = num_clusters;
    wq.cluster_size = (nodes + num_clusters - 1) / num_clusters;  // Round up division
    return wq;
  }

  __host__ static void freeWorkQueue(GlobalWorkQueue& wq) {
    cudaFree(wq.data);
    cudaFree(wq.heads);
    cudaFree(wq.tails);
    cudaFree(wq.locks);
  }

private:
  uint32_t* data;
  uint32_t* heads;
  uint32_t* tails;
  clutra::detail::atomic::TicketLock* locks;
  uint32_t num_clusters;
  uint32_t cluster_size;
};

template <typename GraphDev, size_t BlockSize>
__device__ void processVertex(uint32_t vertex,
                              const GraphDev& graph,
                              uint32_t* distances,
                              LocalHotRing<BlockSize>& local_queue,
                              GlobalWorkQueue& global_queue,
                              size_t cluster_id) {
  auto start = graph.getRowOffsets()[vertex];
  auto end = graph.getRowOffsets()[vertex + 1];
  uint32_t distance_vertex = distances[vertex];

  for (size_t u = start; u < end; u++) {
    auto neighbor = graph.getColumnIndices()[u];
    // if the neighbor is unvisited, set distance and push to local queue
    if (atomicCAS(&distances[neighbor], UINT32_MAX, distance_vertex + 1) == UINT32_MAX) {
      distances[neighbor] = distances[vertex] + 1;

      // Try to push to local queue
      if (!local_queue->push(neighbor)) {
        // Local queue is full, push to global queue
        auto lock = global_queue.getLock(cluster_id);
        lock->acquire();
        uint32_t* tail = global_queue.getTail(cluster_id);
        uint32_t pos = (*tail)++;
        global_queue.getData(cluster_id)[pos] = neighbor;
        lock->release();
      }
    }
  }
}

__device__ bool fetchFromColdBuffer(uint32_t& vertex) {
  return true;
}

template <size_t BlockSize, typename GraphDev>
__global__ void bfsKernel(GraphDev graph, uint32_t* distances, GlobalWorkQueue work_queue, uint32_t clusters) {
  __shared__ CTAHotRings<256, BlockSize> local_rings;
  __shared__ CTAHotRings<256, BlockSize>* cluster_rings[8];

  constexpr uint32_t WARP_SIZE = 32;
  constexpr uint32_t INVALID_VERTEX = UINT32_MAX;
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int lane = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  auto cluster = cg::this_cluster();
  int cluster_id = blockIdx.x / cluster.dim_blocks().x;
  auto& warp_ring = local_rings.getHotRing(warp_id);

  // Initialize local queues
  cluster.sync();
  if (tid == 0) {
    cluster_rings[cluster_id] = &local_rings;
    for (int i = 0; i < clusters; i++) {
      if (i != cluster_id) {
        cluster_rings[i] = cluster.map_shared_rank(&local_rings, i);
      }
    }
  }
  cluster.sync();

  uint32_t vertex = INVALID_VERTEX;
  while (true) {
    if (lane == 0) {
      // Try to pop from local queue
      if (warp_ring.pop(vertex)) {
        printf("Got from active hot ring\n");
      } else if (false) {
        // fetch from local queue
      } else {
        vertex = INVALID_VERTEX;
      }
    }

    vertex = __shfl_sync(__activemask(), vertex, 0);
    if (vertex != INVALID_VERTEX) {
      processVertex(vertex, graph, distances, warp_ring, work_queue, cluster_id);
    } else {
      break;  // No more work in local queue
    }
  }
  cluster.sync();
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
  clutra::stealer::BasicStealer stealer(stealer_config);

  clutra::frontier::FrontierMLB<uint32_t> in_frontier(graph.getVertexCount());
  clutra::frontier::FrontierMLB<uint32_t> out_frontier(graph.getVertexCount());

  if (opts.random_source) {
    opts.source = getRandomSource(graph.getVertexCount());
  }

  int* distances;
  cudaMallocManaged(&distances, graph.getVertexCount() * sizeof(int));
  cudaMemset(distances, -1, graph.getVertexCount() * sizeof(int));

  distances[opts.source] = 0;
  in_frontier.insert(opts.source);

  int iter = 0;

  std::cout << "[*] Running BFS from source vertex " << opts.source << std::endl;

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

  clutra::profile::KernelProfilerManager::instance().printSummary(opts.profiling_detail);
}
