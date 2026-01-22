/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <operators/advance/advance_mlb.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/profile.cuh>
#include <utils/device.cuh>

namespace clutra::operators::advance {

template<clutra::graph::detail::GraphConcept GraphT, typename DerivedStealerT, typename LambdaT>
void push(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, const clutra::stealer::Stealer<DerivedStealerT>& stealer, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, &output_frontier, stealer, std::forward<LambdaT>(functor));
}

template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, &output_frontier, clutra::stealer::NullStealer{}, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, nullptr, clutra::stealer::NullStealer{}, std::forward<LambdaT>(functor));
}

template<clutra::graph::detail::GraphConcept GraphT, typename DerivedStealerT, typename LambdaT>
void pull(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, const clutra::stealer::Stealer<DerivedStealerT>& stealer, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, &output_frontier, stealer, std::forward<LambdaT>(functor));
}

template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void pull(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, clutra::frontier::FrontierMLB<>& output_frontier, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, &output_frontier, clutra::stealer::NullStealer{}, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template<clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void pull(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, nullptr, clutra::stealer::NullStealer{}, std::forward<LambdaT>(functor));
}

}// namespace clutra::operators::advance
