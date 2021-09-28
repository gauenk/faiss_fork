/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/burstblockselect/BurstBlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

#if GPU_MAX_SELECTION_K >= 2048
BURST_BLOCK_SELECT_IMPL(float, false, 2048, 8);
#endif

} // namespace gpu
} // namespace faiss
