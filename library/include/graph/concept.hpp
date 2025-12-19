/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <graph/properties.hpp>

namespace clutra::graph::detail {

template<typename DeviceGraphT>
concept DeviceGraphConcept = requires(DeviceGraphT g) {
  { g.getVertexCount() } -> std::convertible_to<size_t>;
  { g.getEdgeCount() } -> std::convertible_to<size_t>;
  { g.getDegree(std::declval<typename DeviceGraphT::vertex_t>()) } -> std::convertible_to<size_t>;
  { g.getFirstNeighbor(std::declval<typename DeviceGraphT::vertex_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getSourceVertex(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getDestinationVertex(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::vertex_t>;
  { g.getEdgeWeight(std::declval<typename DeviceGraphT::edge_t>()) } -> std::convertible_to<typename DeviceGraphT::weight_t>;
};

template<typename GraphT>
concept GraphConcept = requires(GraphT g) {
  { g.getDeviceGraph() };
  { g.getInverseDeviceGraph() };
  { g.getProperties() } -> std::convertible_to<Properties>;
} && detail::DeviceGraphConcept<GraphT>;

} // namespace detail
