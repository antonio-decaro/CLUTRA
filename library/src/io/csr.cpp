#include <io/csr.hpp>
#include <io/matrix_market.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>


template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> clutra::formats::CSR<ValueT, IndexT, OffsetT>::invert() const {
  IndexT n_rows = getRowOffsetsSize();
  OffsetT n_nonzeros = getNumNonzeros();

  std::vector<OffsetT> inv_row_offsets(static_cast<size_t>(n_rows) + 1, 0);
  std::vector<IndexT> inv_column_indices(n_nonzeros);
  std::vector<ValueT> inv_values(n_nonzeros);

  // Count incoming edges for each column to build row offsets of the inverted matrix
  for (OffsetT idx = 0; idx < n_nonzeros; ++idx) {
    IndexT column = _column_indices[idx];
    inv_row_offsets[static_cast<size_t>(column) + 1]++;
  }

  for (IndexT row = 0; row < n_rows; ++row) { inv_row_offsets[row + 1] += inv_row_offsets[row]; }

  // Insert (col -> row) edges using the computed offsets
  std::vector<OffsetT> write_offsets = inv_row_offsets;
  for (IndexT row = 0; row < n_rows; ++row) {
    OffsetT row_start = _row_offsets[row];
    OffsetT row_end = _row_offsets[row + 1];
    for (OffsetT idx = row_start; idx < row_end; ++idx) {
      IndexT column = _column_indices[idx];
      OffsetT dest = write_offsets[column]++;
      inv_column_indices[dest] = row;
      inv_values[dest] = _nnz_values[idx];
    }
  }

  return CSR<ValueT, IndexT, OffsetT>(std::move(inv_row_offsets), std::move(inv_column_indices), std::move(inv_values));
};

template <typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> clutra::io::csr::fromCSR(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<OffsetT> row_offsets;
  std::vector<IndexT> column_indices;
  std::vector<ValueT> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read row offsets
  for (int i = 0; i < n_rows + 1; ++i) {
    OffsetT offset;
    iss >> offset;
    row_offsets.push_back(offset);
  }

  // Read column indices
  for (int i = 0; i < row_offsets.back(); ++i) {
    IndexT index;
    iss >> index;
    column_indices.push_back(index);
  }

  // Read non-zero values
  for (int i = 0; i < row_offsets.back(); ++i) {
    ValueT value;
    iss >> value;
    nnz_values.push_back(value);
  }

  return clutra::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);

};

template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromMM(std::istream& iss, clutra::graph::Properties* properties) {
  clutra::io::detail::mm::Banner banner;

  size_t rows = 0;
  size_t cols = 0;
  size_t nnz = 0;
  std::vector<std::tuple<IndexT, IndexT, ValueT>> entries;

  // Read matrix dimensions and non-zero count
  std::string line;
  bool dimensions_read = false;
  bool banner_read = false;

  while (std::getline(iss, line)) {
    if (line[0] == '%') {
      if (!banner_read) {
        banner_read = true;
        banner.read(line);
        banner.validate<ValueT, IndexT, OffsetT>();
        if (properties) {
          properties->directed = !banner.isSymmetric();
          properties->weighted = !banner.isPattern();
        }
      }
      continue;
    }; // Skip comments
    std::istringstream line_stream(line);

    if (!dimensions_read) {
      line_stream >> rows >> cols >> nnz;
      dimensions_read = true;
    } else {
      IndexT row;
      IndexT col;
      ValueT value;
      if (banner.isPattern()) {
        line_stream >> row >> col;
        value = static_cast<ValueT>(1);
      } else {
        line_stream >> row >> col >> value;
      }

      entries.emplace_back(row - 1, col - 1, value);

      // For symmetric matrices, also add the transpose entry if not on the diagonal
      if (banner.isSymmetric()) { entries.emplace_back(col - 1, row - 1, value); }
    }
  }

  // Sort entries by row, then by column
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
  });

  // Initialize CSR vectors
  std::vector<OffsetT> row_offsets(rows + 1, 0);
  std::vector<IndexT> column_indices(entries.size());
  std::vector<ValueT> nnz_values(entries.size());

  // Count non-zero elements per row for row_offsets
  for (const auto& entry : entries) { row_offsets[std::get<0>(entry) + 1]++; }

  // Accumulate counts to get row offsets
  for (IndexT i = 1; i <= rows; ++i) { row_offsets[i] += row_offsets[i - 1]; }

  // Fill in column indices and values arrays
  std::vector<OffsetT> row_position(rows, 0);
  for (const auto& entry : entries) {
    IndexT row = std::get<0>(entry);
    IndexT col = std::get<1>(entry);
    ValueT value = std::get<2>(entry);

    OffsetT pos = row_offsets[row] + row_position[row];
    column_indices[pos] = col;
    nnz_values[pos] = value;
    row_position[row]++;
  }

  return clutra::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

template<typename ValueT, typename IndexT, typename OffsetT>
clutra::formats::CSR<ValueT, IndexT, OffsetT> fromBinary(std::istream& iss, clutra::graph::Properties* properties = nullptr) {
  if (!iss) { throw std::runtime_error("Failed to read binary CSR matrix"); }

  size_t num_rows = 0;
  size_t num_nonzero = 0;
  uint64_t maybe_magic = 0;

  iss.read(reinterpret_cast<char*>(&maybe_magic), sizeof(uint64_t));
  if (!iss) { throw std::runtime_error("Failed to read binary CSR matrix"); }

  if (maybe_magic == clutra::io::csr::detail::binary::magic) {
    uint8_t version = 0;
    uint8_t flags = 0;
    uint16_t reserved16 = 0;
    uint32_t reserved32 = 0;

    iss.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    iss.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    iss.read(reinterpret_cast<char*>(&reserved16), sizeof(uint16_t));
    iss.read(reinterpret_cast<char*>(&reserved32), sizeof(uint32_t));
    iss.read(reinterpret_cast<char*>(&num_rows), sizeof(size_t));
    iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));

    if (properties) {
      properties->directed = (flags & clutra::io::csr::detail::binary::directed_mask) != 0;
      properties->weighted = (flags & clutra::io::csr::detail::binary::weighted_mask) != 0;
    }
  } else {
    num_rows = static_cast<size_t>(maybe_magic);
    iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));
    if (properties) {
      properties->directed = true;
      properties->weighted = true;
    }
  }

  std::vector<OffsetT> row_ptr(num_rows);
  std::vector<IndexT> col_indices(num_nonzero);
  std::vector<ValueT> values(num_nonzero);

  iss.read(reinterpret_cast<char*>(row_ptr.data()), row_ptr.size() * sizeof(OffsetT));
  iss.read(reinterpret_cast<char*>(col_indices.data()), col_indices.size() * sizeof(IndexT));
  iss.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(ValueT));

  return {row_ptr, col_indices, values};
}

template<typename ValueT, typename IndexT, typename OffsetT>
void toBinary(const clutra::formats::CSR<ValueT, IndexT, OffsetT>& csr,
              std::ostream& oss,
              const clutra::graph::Properties& properties = clutra::graph::Properties()) {
if (!oss) { throw std::runtime_error("Failed to write binary CSR matrix"); }

  auto& row_offsets = csr.getRowOffsets();
  auto& column_indices = csr.getColumnIndices();
  auto& values = csr.getValues();

  size_t num_rows = row_offsets.size();
  size_t num_nonzero = column_indices.size();

  uint64_t magic = clutra::io::csr::detail::binary::magic;
  uint8_t version = 1;
  uint8_t flags = 0;
  if (properties.directed) { flags |= clutra::io::csr::detail::binary::directed_mask; }
  if (properties.weighted) { flags |= clutra::io::csr::detail::binary::weighted_mask; }
  uint16_t reserved16 = 0;
  uint32_t reserved32 = 0;

  oss.write(reinterpret_cast<const char*>(&magic), sizeof(uint64_t));
  oss.write(reinterpret_cast<const char*>(&version), sizeof(uint8_t));
  oss.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));
  oss.write(reinterpret_cast<const char*>(&reserved16), sizeof(uint16_t));
  oss.write(reinterpret_cast<const char*>(&reserved32), sizeof(uint32_t));

  oss.write(reinterpret_cast<const char*>(&num_rows), sizeof(size_t));
  oss.write(reinterpret_cast<const char*>(&num_nonzero), sizeof(size_t));

  oss.write(reinterpret_cast<const char*>(&row_offsets[0]), row_offsets.size() * sizeof(OffsetT));
  oss.write(reinterpret_cast<const char*>(&column_indices[0]), column_indices.size() * sizeof(IndexT));
  oss.write(reinterpret_cast<const char*>(&values[0]), values.size() * sizeof(ValueT));
}

template class clutra::formats::CSR<float, uint32_t, uint32_t>;
template class clutra::formats::CSR<double, uint32_t, uint32_t>;
template class clutra::formats::CSR<float, uint64_t, uint64_t>;
template class clutra::formats::CSR<double, uint64_t, uint64_t>;