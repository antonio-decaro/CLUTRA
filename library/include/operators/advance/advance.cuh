/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <operators/advance/advance_mlb.cuh>
#include <frontier/frontier.cuh>
#include <utils/profile.cuh>
#include <utils/device.cuh>
#include <concepts>

namespace clutra::operators::advance {

template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph, const clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, LambdaT&& functor) {
  detail::launchKernel(graph, input_frontier, output_frontier, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph, const clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  detail::launchKernel(graph, input_frontier, std::nullopt, std::forward<LambdaT>(functor));
}

template<typename IndexT, typename OffsetT, typename ValueT, typename LambdaT>
void pull(const clutra::graph::GraphCSR<IndexT, OffsetT, ValueT>& graph, const clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, LambdaT&& functor) {
  throw std::runtime_error("Pull advance is not supported for MLB frontiers.");
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template<typename IndexT, typename OffsetT, typename ValueT, typename LambdaT>
void pull(const clutra::graph::GraphCSR<IndexT, OffsetT, ValueT>& graph, const clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  throw std::runtime_error("Pull advance is not supported for MLB frontiers.");
}

}// namespace clutra::operators::advance
