/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/burstblockselect/BurstBlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

BURST_BLOCK_SELECT_IMPL(float, true, 1024, 8);

}
} // namespace faiss
