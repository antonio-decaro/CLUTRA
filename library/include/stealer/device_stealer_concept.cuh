/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace clutra::stealer {

template <typename StealerT>
concept DeviceStealerConcept = requires(StealerT stealer) {

  { stealer.steal() } -> std::same_as<void>;
};

}