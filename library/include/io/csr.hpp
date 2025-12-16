/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <memory>
#include <vector>
#include <graph/graph.cuh>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utils/types.hpp>

namespace clutra {
namespace formats {


/**
 * @class CSR
 * @brief Compressed Sparse Row (CSR) matrix format.
 *
 * This class represents a sparse matrix in CSR format, which is efficient for matrix-vector multiplication.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is types::index_t).
 * @tparam OffsetT Type of the offsets (default is types::offset_t).
 *
 * The CSR format stores the matrix in three separate arrays:
 * - _row_offsets: Array of size (n_rows + 1) that stores the starting index of each row in the _column_indices and _nnz_values arrays.
 * - _column_indices: Array that stores the column indices of the non-zero values.
 * - _nnz_values: Array that stores the non-zero values of the matrix.
 */
template<typename ValueT, typename IndexT = detail::types::index_t, typename OffsetT = detail::types::offset_t>
class CSR {
public:
  /**
   * @brief Default constructor.
   *
   * Creates an empty CSR matrix.
   */
  CSR() = default;


  /**
   * @brief Constructs a CSR (Compressed Sparse Row) matrix.
   *
   * @param row_offsets A vector containing the row offsets.
   * @param column_indices A vector containing the column indices.
   * @param nnz_values A vector containing the non-zero values.
   */
  CSR(std::vector<OffsetT> row_offsets, std::vector<IndexT> column_indices, std::vector<ValueT> nnz_values)
      : _row_offsets(row_offsets), _column_indices(column_indices), _nnz_values(nnz_values) {}

  /**
   * @brief Constructor for the CSR (Compressed Sparse Row) class.
   *
   * This constructor initializes the CSR matrix with the given number of rows and non-zero elements.
   * It resizes the internal vectors to accommodate the specified number of rows and non-zero values.
   *
   * @param n_rows The number of rows in the matrix.
   * @param n_nonzeros The number of non-zero elements in the matrix.
   */
  CSR(IndexT n_rows, OffsetT n_nonzeros) {
    _row_offsets.resize(n_rows + 1);
    _column_indices.resize(n_nonzeros);
    _nnz_values.resize(n_nonzeros);
  }

  /**
   * @brief Default destructor for the CSR class.
   */
  ~CSR() = default;

  // Getters
  /**
   * @brief Get the size of the row offsets.
   *
   * This function returns the size of the row offsets array,
   * which is the number of rows in the CSR (Compressed Sparse Row) format matrix.
   *
   * @return IndexT The size of the row offsets array minus one.
   */
  IndexT getRowOffsetsSize() const { return _row_offsets.size() - 1; }

  /**
   * @brief Returns the number of non-zero elements in the CSR (Compressed Sparse Row) matrix.
   *
   * This function calculates the number of non-zero elements by returning the size of the
   * column_indices vector, which stores the column indices of the non-zero elements.
   *
   * @return OffsetT The number of non-zero elements in the CSR matrix.
   */
  OffsetT getNumNonzeros() const { return _column_indices.size(); }

  /**
   * @brief Retrieves the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function returns a constant reference to the vector containing the row offsets.
   * The row offsets vector indicates the starting index of each row in the values array.
   *
   * @return const std::vector<OffsetT>& A constant reference to the vector of row offsets.
   */
  const std::vector<OffsetT>& getRowOffsets() const { return _row_offsets; }

  /**
   * @brief Retrieves the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function returns a reference to the vector containing the row offsets.
   * The row offsets vector indicates the starting index of each row in the values array.
   *
   * @return std::vector<OffsetT>& A constant reference to the vector of row offsets.
   */
  std::vector<OffsetT>& getRowOffsets() { return _row_offsets; }

  /**
   * @brief Retrieves the column indices of the CSR (Compressed Sparse Row) format.
   *
   * @return A constant reference to a vector containing the column indices.
   */
  const std::vector<IndexT>& getColumnIndices() const { return _column_indices; }

  /**
   * @brief Retrieves the column indices of the CSR (Compressed Sparse Row) format.
   *
   * @return A reference to a vector containing the column indices.
   */
  std::vector<IndexT>& getColumnIndices() { return _column_indices; }

  /**
   * @brief Retrieves the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * @return A constant reference to a vector containing the non-zero values.
   */
  const std::vector<ValueT>& getValues() const { return _nnz_values; }

  /**
   * @brief Retrieves the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * @return A reference to a vector containing the non-zero values.
   */
  std::vector<ValueT>& getValues() { return _nnz_values; }

  // Setters

  /**
   * @brief Sets the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the row offsets of the CSR matrix to the specified vector of offsets.
   *
   * @param offsets A vector containing the row offsets.
   */
  void setRowOffsets(const std::vector<OffsetT>& offsets) { _row_offsets = offsets; }

  /**
   * @brief Sets the column indices of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the column indices of the CSR matrix to the specified vector of indices.
   *
   * @param indices A vector containing the column indices.
   */
  void setColumnIndices(const std::vector<IndexT>& indices) { _column_indices = indices; }

  /**
   * @brief Sets the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the non-zero values of the CSR matrix to the specified vector of values.
   *
   * @param values A vector containing the non-zero values.
   */
  void setNnzValues(const std::vector<ValueT>& values) { _nnz_values = values; }

  /**
   * @brief Returns the transpose (inverse adjacency) of the current CSR matrix.
   *
   * This method constructs a new CSR where rows and columns are swapped. It is mainly used to build
   * the incoming-edge structure of a directed graph starting from its outgoing adjacency.
   *
   * @return CSR<ValueT, IndexT, OffsetT> The transposed CSR matrix.
   */
  CSR<ValueT, IndexT, OffsetT> invert() const;

private:
  std::vector<OffsetT> _row_offsets;
  std::vector<IndexT> _column_indices;
  std::vector<ValueT> _nnz_values;
};

} // namespace formats

namespace io::csr {
namespace detail::binary {
static constexpr uint64_t magic = 0x4342535200000001ULL; // "CBSR" + version marker
static constexpr uint8_t directed_mask = 0x1;
static constexpr uint8_t weighted_mask = 0x2;
} // namespace detail

/**
 * @brief Converts a matrix in CSR format from a file to a CSR object.
 * This function reads a matrix in CSR format from a given input stream and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix
 * 
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param iss The input stream containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromCSR(std::istream& iss);

/**
 * @brief Reads a Matrix Market file in coordinate format and converts it to a CSR matrix.
 *
 * This function parses a Matrix Market file from an input stream and converts it to a CSR (Compressed Sparse Row)
 * matrix. It also optionally fills in the graph properties based on the Matrix Market banner information.
 *
 * @tparam ValueT The type of the values in the matrix.
 * @tparam IndexT The type of the indices in the matrix.
 * @tparam OffsetT The type of the offsets in the CSR format.
 * @param iss The input stream containing the Matrix Market file.
 * @param properties Optional pointer to a Properties structure to store graph properties.
 * @return A CSR formatted sparse matrix.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromMM(std::istream& iss, clutra::graph::Properties* properties = nullptr);

/**
 * @brief Converts a matrix in CSR format from a file to a CSR object.
 *
 * This function reads a matrix in CSR format from a given file and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix.
 *
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param filename The name of the file containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 * @throws std::runtime_error if the file fails to open.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromMM(const std::string& filename, clutra::graph::Properties* properties = nullptr) {
  std::ifstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Failed to open file: " + filename); }

  return fromMM<ValueT, IndexT, OffsetT>(file, properties);
}

/**
 * @brief Reads a CSR (Compressed Sparse Row) matrix from a binary input stream.
 *
 * This function reads the number of rows and non-zero elements from the binary
 * input stream, followed by the row pointers, column indices, and values arrays.
 * It then constructs and returns a CSR matrix using these arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row pointers in the CSR matrix.
 * @param iss The input stream to read the binary data from.
 * @return A CSR matrix containing the data read from the input stream.
 * @throws std::runtime_error If the input stream is not valid or if reading fails.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromBinary(std::istream& iss, clutra::graph::Properties* properties = nullptr);

/**
 * @brief Serializes a CSR (Compressed Sparse Row) matrix to a binary stream.
 *
 * This function writes the CSR matrix data to the provided output stream in binary format.
 * The CSR matrix is represented by its row offsets, column indices, and values arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row offsets in the CSR matrix.
 * @param csr The CSR matrix to be serialized.
 * @param oss The output stream to which the CSR matrix will be written.
 *
 * @throws std::runtime_error If the output stream is not in a good state.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
void toBinary(const clutra::formats::CSR<ValueT, IndexT, OffsetT>& csr,
              std::ostream& oss,
              const clutra::graph::Properties& properties = clutra::graph::Properties());

} // namespace io::csr
} // namespace clutra