/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"
#include "utils.cuh"
#include <algorithm>
#include <cuda.h>

namespace clutra::frontier {

namespace detail {

template<typename T, size_t Levels, typename B = clutra::detail::types::bitmap_type_t>
class MLBDevice {
public:
  using bitmap_type = B;
  static constexpr uint32_t alignment = 32; ///< Number of elements for alignment (warp sized).

  static constexpr uint32_t align_up(uint32_t value) {
    return (value + alignment - 1) / alignment * alignment;
  }

  MLBDevice(size_t num_elems) : _num_elems(num_elems) {
    _range = sizeof(bitmap_type) * clutra::detail::types::byte_size;
    _size[0] = (num_elems / _range) + (num_elems % _range != 0 ? 1 : 0);
    _size[0] = align_up(_size[0]);

    for (uint16_t i = 1; i < Levels; i++) {
      _size[i] = (_size[i - 1] / _range) + (_size[i - 1] % _range != 0 ? 1 : 0);
      _size[i] = align_up(std::max<uint32_t>(_size[i], 1));
    }
  }

  __host__ __device__ inline uint32_t getBitmapSize() const { return _size[0]; }

  __host__ __device__ inline uint32_t getNumElems() const { return _num_elems; }

  __host__ __device__ inline uint32_t getBitmapRange() const { return _range; }

  __host__ __device__ inline bitmap_type* getData() const { return _data[0]; }

  __device__ inline void set(uint32_t idx, bool val) const {
    if (val) {
      insert(idx);
    } else {
      remove(idx);
    }
  }

  __device__ inline bool insert(T idx) const {
#pragma unroll
    for (uint16_t i = 0; i < Levels; i++) {
      T lidx = idx;
      for (uint16_t _ = 0; _ < i; _++) { lidx /= _range; } // the index must be divided by the range^level
      if (!(_data[i][getBitmapIndex(lidx)] & (static_cast<bitmap_type>(1) << (lidx % _range)))) {
        atomicOr(&_data[i][getBitmapIndex(lidx)], static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (lidx % _range)));
      }
    }
    return true;
  }

  __device__ inline bool remove(uint32_t idx) const {
    atomicAnd(&_data[0][getBitmapIndex(idx)], ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % _range))));
    for (uint16_t i = 1; i < Levels; i++) {
      uint32_t lidx = idx;
      for (uint16_t _ = 0; _ < i; _++) { lidx /= _range; } // the index must be divided by the range^level
      atomicAnd(&_data[i][getBitmapIndex(lidx)], ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (lidx % _range))));
    }
    return true;
  }

  __device__ inline void reset() const {
    for (uint16_t i = 0; i < _size; i++) { _data[i] = static_cast<bitmap_type>(0); }
  }

  __device__ inline void reset(uint32_t id) const { _data[id] = static_cast<bitmap_type>(0); }

  __device__ inline bool check(uint32_t idx) const { return _data[0][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range)); }

  __device__ inline bool empty() const {
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < _size[Levels - 1]; i++) { count += _data[Levels - 1][i]; }
    return count == static_cast<bitmap_type>(0);
  }

  __device__ inline bool empty(uint32_t el_idx, uint16_t level) const { return _data[level][el_idx]; }

  __host__ __device__ inline uint32_t getBitmapIndex(uint32_t idx) const { return idx / _range; }

  __host__ __device__ inline int* getOffsets() const { return _offsets; }

  __host__ __device__ inline uint32_t* getOffsetsSize() const { return _offsets_size; }

  __host__ __device__ inline uint32_t getBitmapSize(const uint level) const { return _size[level]; }

  __host__ __device__ inline bitmap_type* getData(const uint level) const { return _data[level]; }

  __device__ inline bool check(const uint level, uint32_t idx) const {
    return _data[level][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range));
  }

  __host__ void setData(bitmap_type* data[Levels]) {
    for (uint16_t i = 0; i < Levels; i++) { this->_data[i] = data[i]; }
  }

  __host__ void setOffsets(int* offsets) { this->_offsets = offsets; }

  __host__ void setOffsetsSize(uint32_t* offsets_size) { this->_offsets_size = offsets_size; }

protected:
  uint _range;                ///< The range of the bitmap.
  uint32_t _num_elems;        ///< The number of elements in the bitmap.
  uint32_t _size[Levels];     ///< The size of the bitmap.
  bitmap_type* _data[Levels]; ///< Pointer to the bitmap.

  int* _offsets;
  uint32_t* _offsets_size;
};

} // namespace detail 

template<typename T, size_t Levels = 2>
class FrontierMLB {
public:
  using bitmap_type = typename detail::MLBDevice<T, Levels>::bitmap_type;
  using DeviceFrontier = detail::MLBDevice<T, Levels>;

  FrontierMLB(size_t num_elems);
  ~FrontierMLB();

  size_t getBitmapSize() const {return _bitmap.getBitmapSize();}
  size_t getNumElems() const {return _bitmap.getNumElems();}
  size_t getBitmapRange() const {return _bitmap.getBitmapRange();}

  inline bool selfAllocated() const {return false;}
  bool empty() const;
  bool check(size_t idx) const;
  bool insert(size_t idx);
  bool remove(size_t idx);
  size_t size() const;

  FrontierMLB& operator=(const FrontierMLB& other);
  void merge(FrontierMLB<T>& other);
  void intersect(FrontierMLB<T>& other);
  void clear();

  const DeviceFrontier& getDeviceFrontier() const { return _bitmap; }
  void computeActiveFrontier(bool invert = false) const;
  size_t getActiveFrontierSize() const;

protected:
  DeviceFrontier _bitmap; ///< The bitmap.
  uint32_t* _host_offsets_size = nullptr; ///< Host-pinned mirror of offsets_size.
};


} // namespace clutra::frontier


//   sygraph::frontier::detail::BitmapState<Levels, bitmap_type> saveState() {
//     sygraph::frontier::detail::BitmapState<Levels, bitmap_type> state;
// #pragma unroll
//     for (size_t i = 0; i < Levels; i++) {
//       state.size[i] = _bitmap.getBitmapSize(i);
//       state.data[i].resize(state.size[i]);
//       auto e = _queue.copy(_bitmap.getData(i), state.data[i].data(), state.size[i]);
//     }

//     _queue.wait();
//     return state;
//   }

//   void loadState(const sygraph::frontier::detail::BitmapState<Levels, bitmap_type>& state) {
// #pragma unroll
//     for (size_t i = 0; i < Levels; i++) {
//       assert(state.size[i] == _bitmap.getBitmapSize(i));
//       auto e = _queue.copy(state.data[i].data(), _bitmap.getData(i), state.size[i]);
//     }
//     _queue.wait();
//   }
