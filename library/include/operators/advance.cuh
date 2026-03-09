/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/concept.hpp>
#include <graph/graph.cuh>
#include <operators/advance/advance_mlb.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/profile.cuh>

namespace clutra::operators::advance {

/**
 * @brief Advance operator in push mode with input and output frontiers.
 * @param graph The input graph.
 * @param input_frontier The input frontier containing active vertices.
 * @param output_frontier The output frontier to store active vertices after the advance.
 * @param stealer The stealer configuration for load balancing.
 * @param functor The user-defined functor to apply during the advance.
 */
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
          load_balance load_balance,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, &output_frontier, stealer, load_balance,
                                                        std::forward<LambdaT>(functor));
}

template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
          LambdaT&& functor) {
  push(graph, input_frontier, output_frontier, stealer, load_balance::bucketing, std::forward<LambdaT>(functor));
}

template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          load_balance load_balance,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, &output_frontier,
                                                        clutra::stealer::NullStealer{}, load_balance,
                                                        std::forward<LambdaT>(functor));
}

// Overload for advance in push mode without stealer (uses NullStealer).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          LambdaT&& functor) {
  push(graph, input_frontier, output_frontier, load_balance::bucketing, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          load_balance load_balance,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, nullptr, clutra::stealer::NullStealer{},
                                                        load_balance, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void push(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  push(graph, input_frontier, load_balance::bucketing, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
          load_balance load_balance,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, input_frontier, nullptr, stealer, load_balance,
                                                        std::forward<LambdaT>(functor));
}

template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void push(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
          LambdaT&& functor) {
  push(graph, input_frontier, stealer, load_balance::bucketing, std::forward<LambdaT>(functor));
}

/**
 * @brief Advance operator in pull mode with input and output frontiers.
 * @param graph The input graph.
 * @param input_frontier The input frontier containing active vertices.
 * @param output_frontier The output frontier to store active vertices after the advance.
 * @param stealer The stealer configuration for load balancing.
 * @param functor The user-defined functor to apply during the advance.
 */
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void pull(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, &output_frontier, stealer,
                                                        std::forward<LambdaT>(functor));
}

// Overload for advance in pull mode without stealer (uses NullStealer).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void pull(const GraphT& graph,
          clutra::frontier::FrontierMLB<>& input_frontier,
          clutra::frontier::FrontierMLB<>& output_frontier,
          LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, &output_frontier,
                                                        clutra::stealer::NullStealer{}, std::forward<LambdaT>(functor));
}

// Overload for advance when no output frontier is needed (e.g., counting-only or side-effect functors).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void pull(const GraphT& graph, clutra::frontier::FrontierMLB<>& input_frontier, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::pull>(graph, input_frontier, nullptr, clutra::stealer::NullStealer{},
                                                        std::forward<LambdaT>(functor));
}

/**
 * Performs a graph advance operation in push mode starting from all vertices in the graph.
 * No input frontier is used; all vertices are considered active.
 * @param graph The input graph.
 * @param output_frontier The output frontier to store active vertices after the advance.
 * @param stealer The stealer configuration for load balancing.
 * @param functor The user-defined functor to apply during the advance.
 */
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void graph(const GraphT& graph,
           clutra::frontier::FrontierMLB<>& output_frontier,
           const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
           load_balance load_balance,
           LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, &output_frontier, stealer, load_balance,
                                                        std::forward<LambdaT>(functor));
}

template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void graph(const GraphT& graph,
           clutra::frontier::FrontierMLB<>& output_frontier,
           const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
           LambdaT&& functor) {
  clutra::operators::advance::graph(graph, output_frontier, stealer, load_balance::bucketing,
                                    std::forward<LambdaT>(functor));
}

// Overload for advance in push mode without output frontier (discards results).
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void graph(const GraphT& graph,
           const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
           load_balance load_balance,
           LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, nullptr, stealer, load_balance,
                                                        std::forward<LambdaT>(functor));
}

// Overload for advance in push mode without output frontier (discards results).
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void graph(const GraphT& graph,
           const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
           LambdaT&& functor) {
  clutra::operators::advance::graph(graph, stealer, load_balance::bucketing, std::forward<LambdaT>(functor));
}

// Overload for advance in push mode without stealer (uses NullStealer) and with output frontier.
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void graph(const GraphT& graph, load_balance load_balance, LambdaT&& functor) {
  detail::launchKernel<detail::advance_direction::push>(graph, nullptr, clutra::stealer::NullStealer{}, load_balance,
                                                        std::forward<LambdaT>(functor));
}

// Overload for advance in push mode without stealer (uses NullStealer).
template <clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void graph(const GraphT& graph, LambdaT&& functor) {
  clutra::operators::advance::graph(graph, load_balance::bucketing, std::forward<LambdaT>(functor));
}

}  // namespace clutra::operators::advance
