/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/properties.hpp>
#include <io/csr.hpp>

namespace clutra::graph {

namespace detail {

template<typename IndexT, typename OffsetT, typename ValueT>
class GraphCSRDevice {
public:
  using vertex_t = IndexT; ///< The type used to represent vertices of the graph.
  using edge_t = OffsetT;  ///< The type used to represent edges of the graph.
  using weight_t = ValueT; ///< The type used to represent weights of the graph.
  struct NeighborIterator {
    __device__ NeighborIterator(IndexT* start_ptr, IndexT* ptr) : _start_ptr(start_ptr), _ptr(ptr) {}

    __device__ inline IndexT operator*() const { return *_ptr; }

    __device__ inline NeighborIterator& operator++() {
      ++_ptr;
      return *this;
    }

    __device__ inline NeighborIterator operator+(int n) const {
      NeighborIterator tmp = *this;
      tmp._ptr += n;
      return tmp;
    }

    __device__ inline bool operator==(const NeighborIterator& other) const { return _ptr == other._ptr; }

    __device__ inline bool operator!=(const NeighborIterator& other) const { return _ptr != other._ptr; }

    __device__ inline edge_t getIndex() const { return static_cast<edge_t>(_ptr - _start_ptr); }

    IndexT* _ptr;
    IndexT* _start_ptr;
  };

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  __host__ __device__ inline size_t getVertexCount() const { return _n_rows; }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  __host__ __device__ inline size_t getEdgeCount() const { return _n_nonzeros; }

  /**
   * @brief Returns the number of neighbors of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbors.
   */
  __device__ inline size_t getDegree(vertex_t vertex) const { return _row_offsets[vertex + 1] - _row_offsets[vertex]; }

  /**
   * @brief Returns the index of the first neighbor of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbor.
   */
  __device__ inline vertex_t getFirstNeighbor(vertex_t vertex) const { return _row_offsets[vertex]; }

  // getters
  __host__ __device__ IndexT* getColumnIndices() const { return _column_indices; }

  __host__ __device__ OffsetT* getRowOffsets() const { return _row_offsets; }

  __host__ __device__ ValueT* getValues() const { return _nnz_values; }

  __device__ vertex_t getSourceVertex(edge_t edge) const {
    // binary search
    vertex_t low = 0;
    vertex_t high = _n_rows - 1;
    while (low <= high) {
      vertex_t mid = low + (high - low) / 2;
      if (_row_offsets[mid] <= edge && edge < _row_offsets[mid + 1]) {
        return mid;
      } else if (_row_offsets[mid] > edge) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return _n_rows;
  }

  __device__ vertex_t getDestinationVertex(edge_t edge) const { return _column_indices[edge]; }

  __device__ weight_t getEdgeWeight(edge_t edge) const { return _nnz_values[edge]; }

  __device__ inline GraphCSRDevice::NeighborIterator begin(vertex_t vertex) const {
    return NeighborIterator(_column_indices, _column_indices + _row_offsets[vertex]);
  }

  __device__ inline GraphCSRDevice::NeighborIterator end(vertex_t vertex) const {
    return NeighborIterator(_column_indices, _column_indices + _row_offsets[vertex + 1]);
  }

  IndexT _n_rows;      ///< The number of rows in the graph.
  OffsetT _n_nonzeros; ///< The number of non-zero values in the graph.

  IndexT* _column_indices; ///< Pointer to the column indices of the graph.
  OffsetT* _row_offsets;   ///< Pointer to the row offsets of the graph.
  ValueT* _nnz_values;     ///< Pointer to the non-zero values of the graph.
};

} // namespace detail


/**
 * @class graph_csr_t
 * @brief Represents a graph in Compressed Sparse Row (CSR) format.
 * @tparam index_t The type used to represent indices of the graph.
 * @tparam offset_t The type used to represent offsets of the graph.
 * @tparam value_t The type used to represent values of the graph.
 */
template<typename IndexT, typename OffsetT, typename ValueT>
class GraphCSR {
public:
  using vertex_t = IndexT; ///< The type used to represent vertices of the graph.
  using edge_t = OffsetT;  ///< The type used to represent edges of the graph.
  using weight_t = ValueT; ///< The type used to represent weights of the graph.

  /**
   * @brief Constructs a graph_csr_t object.
   * @param q The SYCL queue to be used for memory operations.
   * @param csr The CSR format of the graph.
   * @param properties The properties of the graph.
   */
  GraphCSR(clutra::formats::CSR<ValueT, IndexT, OffsetT>& csr, Properties properties);
  GraphCSR(GraphCSR&& other) noexcept;
  GraphCSR(const GraphCSR&) = delete;
  GraphCSR& operator=(const GraphCSR&) = delete;
  GraphCSR& operator=(GraphCSR&&) = delete;

  /**
   * @brief Destroys the graph_csr_t object and frees the allocated memory.
   */
  ~GraphCSR();

  /* Methods */

  const auto& getDeviceGraph() const { return _device_graph; }

  const auto& getTransposedDeviceGraph() const { return _inverse_device_graph; }

  /* Override superclass methods */

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  size_t getVertexCount() const;

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  size_t getEdgeCount() const;

  /**
   * @brief Returns the number of neighbors (out degree) of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbors.
   */
  size_t getDegree(vertex_t vertex) const;

  /**
   * @brief Returns the index of the first neighbor of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbor.
   */
  vertex_t getFirstNeighbor(vertex_t vertex) const;

  vertex_t getSourceVertex(edge_t edge) const;

  vertex_t getDestinationVertex(edge_t edge) const;

  weight_t getEdgeWeight(edge_t edge) const;

  /* Getters and Setters for CSR Graph */

  /**
   * @brief Returns the number of rows in the graph.
   * @return The number of rows.
   */
  IndexT getOffsetsSize() const { return _device_graph.getVertexCount() + 1; }

  /**
   * @brief Returns the number of non-zero values in the graph.
   * @return The number of non-zero values.
   */
  OffsetT getValuesSize() const { return _device_graph.getEdgeCount(); }

  /**
   * @brief Returns a constant pointer to the column indices of the graph.
   * @return A constant pointer to the column indices.
   */
  const IndexT* getColumnIndices() const { return _csr.getColumnIndices().data(); }

  /**
   * @brief Returns a constant pointer to the row offsets of the graph.
   * @return A constant pointer to the row offsets.
   */
  const OffsetT* getRowOffsets() const { return _csr.getRowOffsets().data(); }


  /**
   * @brief Returns a constant pointer to the non-zero values of the graph.
   * @return A constant pointer to the non-zero values.
   */
  const ValueT* getValues() const { return _csr.getValues().data(); }

  Properties getProperties() const { return _properties; }

private:
  Properties _properties;
  const clutra::formats::CSR<ValueT, IndexT, OffsetT>& _csr;
  detail::GraphCSRDevice<IndexT, OffsetT, ValueT> _device_graph;
  detail::GraphCSRDevice<IndexT, OffsetT, ValueT> _inverse_device_graph;
};

template<typename IndexT, typename OffsetT, typename ValueT>
auto createGraph(clutra::formats::CSR<ValueT, IndexT, OffsetT>& csr, Properties properties) {
  return GraphCSR<IndexT, OffsetT, ValueT>(csr, properties);
}

}
