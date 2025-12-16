/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"
#include "utils.cuh"
#include <cuda.h>

namespace clutra::frontier {

namespace detail {

template<typename T, size_t Levels, typename B = clutra::detail::types::bitmap_type_t>
class MLBDevice {
public:
  using bitmap_type = B;

  MLBDevice(size_t num_elems) : _num_elems(num_elems) {
    _range = sizeof(bitmap_type) * clutra::detail::types::byte_size;
    _size[0] = (num_elems / _range) + (num_elems % _range != 0 ? 1 : 0);

    for (uint16_t i = 1; i < Levels; i++) { _size[i] = (_size[i - 1] / _range) + (_size[i - 1] % _range != 0 ? 1 : 0); }
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

  const DeviceFrontier& getDeviceFrontier() const;
  size_t computeActiveFrontier(bool invert = false) const;

protected:
  DeviceFrontier _bitmap; ///< The bitmap.
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

  // /**
  //  * @brief Computes the active frontier by populating the offsets array with the indices of active elements.
  //  * @param invert If true, computes the inactive frontier instead (for pull-based advance operations).
  //  */
  // sycl::event computeActiveFrontier(bool invert = false) const {
  //   sycl::range<1> local_range{types::detail::COMPUTE_UNIT_SIZE};
  //   auto bitmap = this->getDeviceFrontier();
  //   size_t size = bitmap.getBitmapSize(1);
  //   uint32_t range = bitmap.getBitmapRange();
  //   // sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};
  //   size_t global_size = sygraph::detail::device::getNumComputeUnits(_queue) * local_range[0];
  //   sycl::range<1> global_range{global_size};

  //   auto e = this->_queue.submit([&](sycl::handler& cgh) {
  //     sycl::local_accessor<int, 1> local_offsets(local_range[0] * range, cgh);
  //     sycl::local_accessor<uint32_t, 1> local_size(1, cgh);

  //     cgh.parallel_for<mlb_compute_active_frontier_kernel>(
  //         sycl::nd_range<1>{global_range, local_range},
  //         [=, offsets_size = bitmap.getOffsetsSize(), offsets = bitmap.getOffsets()](sycl::nd_item<1> item) {
  //           // if (offsets_size[0] > 0) { return; } // TODO optimize for multiple calls on the same frontier
  //           int gid = item.get_global_linear_id();
  //           auto group = item.get_group();
  //           if (item.get_global_linear_id() == 0) { offsets_size[0] = 0; }
  //           sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> local_size_ref(local_size[0]);
  //           sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

  //           if (group.leader()) { local_size_ref.store(0); }
  //           sycl::group_barrier(group);
  //           for (uint32_t gid = item.get_global_linear_id(); gid < size; gid += item.get_global_range(0)) {
  //             bitmap_type data = bitmap.getData(1)[gid];
  //             for (size_t i = 0; i < range; i++) {
  //               bool is_active = (data & (static_cast<bitmap_type>(1) << i)) != 0;
  //               uint32_t pos;
  //               if ((!invert && !is_active)
  //                   || (invert && (is_active && bitmap.getData(0)[i + gid * range] == std::numeric_limits<bitmap_type>::max()))) {
  //                 continue;
  //               }

  //               local_offsets[local_size_ref++] = static_cast<int>(i + gid * range);
  //             }
  //           }

  //           sycl::group_barrier(group);

  //           size_t data_offset = 0;
  //           if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
  //           data_offset = sycl::group_broadcast(group, data_offset, 0);
  //           for (size_t i = item.get_local_linear_id(); i < local_size_ref.load(); i += item.get_local_range(0)) {
  //             offsets[data_offset + i] = local_offsets[i];
  //           }
  //         });
  //   });

  //   return e;
  // }
