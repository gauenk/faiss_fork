/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/burstblockselect/BurstBlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

BURST_BLOCK_SELECT_IMPL(float, true, 1, 1);
BURST_BLOCK_SELECT_IMPL(float, false, 1, 1);

} // namespace gpu
} // namespace faiss
