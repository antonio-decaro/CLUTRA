/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/graph.cuh>
#include <utils/misc.cuh>

using namespace clutra::graph;

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::GraphCSR(clutra::formats::CSR<ValueT, IndexT, OffsetT>& csr, Properties properties)
    : _csr(csr), _properties(properties) {
  IndexT n_rows = csr.getRowOffsetsSize();
  OffsetT n_nonzeros = csr.getNumNonzeros();
  IndexT* row_offsets;
  OffsetT* column_indices;
  ValueT* nnz_values;
  CUDA_CHECK(cudaMalloc(&row_offsets, (n_rows + 1) * sizeof(IndexT)));
  CUDA_CHECK(cudaMalloc(&column_indices, n_nonzeros * sizeof(OffsetT)));
  CUDA_CHECK(cudaMalloc(&nnz_values, n_nonzeros * sizeof(ValueT)));

  CUDA_CHECK(cudaMemcpy(row_offsets, csr.getRowOffsets().data(), (n_rows + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(column_indices, csr.getColumnIndices().data(), n_nonzeros * sizeof(OffsetT), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(nnz_values, csr.getValues().data(), n_nonzeros * sizeof(ValueT), cudaMemcpyHostToDevice));

  this->_device_graph = {n_rows, n_nonzeros, column_indices, row_offsets, nnz_values};

  if (properties.directed) {
    auto inverted_csr = csr.invert();

    IndexT* inv_row_offsets;
    OffsetT* inv_column_indices;
    ValueT* inv_nnz_values;
    CUDA_CHECK(cudaMalloc(&inv_row_offsets, (n_rows + 1) * sizeof(IndexT)));
    CUDA_CHECK(cudaMalloc(&inv_column_indices, n_nonzeros * sizeof(OffsetT)));
    CUDA_CHECK(cudaMalloc(&inv_nnz_values, n_nonzeros * sizeof(ValueT)));

    CUDA_CHECK(cudaMemcpy(inv_row_offsets, inverted_csr.getRowOffsets().data(), (n_rows + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inv_column_indices, inverted_csr.getColumnIndices().data(), n_nonzeros * sizeof(OffsetT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inv_nnz_values, inverted_csr.getValues().data(), n_nonzeros * sizeof(ValueT), cudaMemcpyHostToDevice));

    this->_inverse_device_graph = {n_rows, n_nonzeros, inv_column_indices, inv_row_offsets, inv_nnz_values};
  } else {
    this->_inverse_device_graph = this->_device_graph;
  }
}

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::~GraphCSR() {
  if (_device_graph._row_offsets != nullptr) {
    CUDA_CHECK(cudaFree(_device_graph._row_offsets));
    _device_graph._row_offsets = nullptr;
  }
  if (_device_graph._column_indices != nullptr) {
    CUDA_CHECK(cudaFree(_device_graph._column_indices));
    _device_graph._column_indices = nullptr;
  }
  if (_device_graph._nnz_values != nullptr) {
    CUDA_CHECK(cudaFree(_device_graph._nnz_values));
    _device_graph._nnz_values = nullptr;
  }
  if (_properties.directed) {
    if (_inverse_device_graph._row_offsets != nullptr) {
      CUDA_CHECK(cudaFree(_inverse_device_graph._row_offsets));
      _inverse_device_graph._row_offsets = nullptr;
    }
    if (_inverse_device_graph._column_indices != nullptr) {
      CUDA_CHECK(cudaFree(_inverse_device_graph._column_indices));
      _inverse_device_graph._column_indices = nullptr;
    }
    if (_inverse_device_graph._nnz_values != nullptr) {
      CUDA_CHECK(cudaFree(_inverse_device_graph._nnz_values));
      _inverse_device_graph._nnz_values = nullptr;
    }
  }
}


template<typename IndexT, typename OffsetT, typename ValueT>
size_t GraphCSR<IndexT, OffsetT, ValueT>::getVertexCount() const { return _device_graph.getVertexCount(); }

template<typename IndexT, typename OffsetT, typename ValueT>
size_t GraphCSR<IndexT, OffsetT, ValueT>::getEdgeCount() const { return _device_graph.getEdgeCount(); }

template<typename IndexT, typename OffsetT, typename ValueT>
size_t GraphCSR<IndexT, OffsetT, ValueT>::getDegree(vertex_t vertex) const {
  return _csr.getRowOffsets()[vertex + 1] - _csr.getRowOffsets()[vertex];
}

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::vertex_t GraphCSR<IndexT, OffsetT, ValueT>::getFirstNeighbor(vertex_t vertex) const {
  return _csr.getRowOffsets()[vertex];
}

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::vertex_t GraphCSR<IndexT, OffsetT, ValueT>::getSourceVertex(edge_t edge) const {
  // binary search
  vertex_t low = 0;
  vertex_t high = _csr.getRowOffsetsSize() - 1;
  while (low <= high) {
    vertex_t mid = low + ((high - low) / 2);
    if (_csr.getRowOffsets()[mid] <= edge && edge < _csr.getRowOffsets()[mid + 1]) {
      return mid;
    } 
    if (_csr.getRowOffsets()[mid] > edge) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return _csr.getRowOffsetsSize();
}

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::vertex_t GraphCSR<IndexT, OffsetT, ValueT>::getDestinationVertex(edge_t edge) const {
  return _csr.getColumnIndices()[edge];
}

template<typename IndexT, typename OffsetT, typename ValueT>
GraphCSR<IndexT, OffsetT, ValueT>::weight_t GraphCSR<IndexT, OffsetT, ValueT>::getEdgeWeight(edge_t edge) const {
  return _csr.getValues()[edge];
}

template class GraphCSR<uint32_t, uint32_t, float>;
template class GraphCSR<uint32_t, uint32_t, double>;
template class GraphCSR<uint32_t, uint32_t, uint32_t>;
template class GraphCSR<uint32_t, uint32_t, uint64_t>;
template class GraphCSR<uint32_t, uint32_t, uint16_t>;