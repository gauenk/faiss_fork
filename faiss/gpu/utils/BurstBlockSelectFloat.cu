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

// warp Q to thread Q:
// 1, 1
// 32, 2
// 64, 3
// 128, 3
// 256, 4
// 512, 8
// 1024, 8
// 2048, 8

BURST_BLOCK_SELECT_DECL(float, true, 1);
BURST_BLOCK_SELECT_DECL(float, true, 32);
BURST_BLOCK_SELECT_DECL(float, true, 64);
BURST_BLOCK_SELECT_DECL(float, true, 128);
BURST_BLOCK_SELECT_DECL(float, true, 256);
BURST_BLOCK_SELECT_DECL(float, true, 512);
BURST_BLOCK_SELECT_DECL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BURST_BLOCK_SELECT_DECL(float, true, 2048);
#endif

BURST_BLOCK_SELECT_DECL(float, false, 1);
BURST_BLOCK_SELECT_DECL(float, false, 32);
BURST_BLOCK_SELECT_DECL(float, false, 64);
BURST_BLOCK_SELECT_DECL(float, false, 128);
BURST_BLOCK_SELECT_DECL(float, false, 256);
BURST_BLOCK_SELECT_DECL(float, false, 512);
BURST_BLOCK_SELECT_DECL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BURST_BLOCK_SELECT_DECL(float, false, 2048);
#endif

void runBurstBlockSelect(
        Tensor<float, 3, true>& in,
        Tensor<float, 3, true>& outK,
        Tensor<int, 3, true>& outV,
        bool dir,
        int k,
        cudaStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BURST_BLOCK_SELECT_CALL(float, true, 1);
        } else if (k <= 32) {
            BURST_BLOCK_SELECT_CALL(float, true, 32);
        } else if (k <= 64) {
            BURST_BLOCK_SELECT_CALL(float, true, 64);
        } else if (k <= 128) {
            BURST_BLOCK_SELECT_CALL(float, true, 128);
        } else if (k <= 256) {
            BURST_BLOCK_SELECT_CALL(float, true, 256);
        } else if (k <= 512) {
            BURST_BLOCK_SELECT_CALL(float, true, 512);
        } else if (k <= 1024) {
            BURST_BLOCK_SELECT_CALL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BURST_BLOCK_SELECT_CALL(float, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BURST_BLOCK_SELECT_CALL(float, false, 1);
        } else if (k <= 32) {
            BURST_BLOCK_SELECT_CALL(float, false, 32);
        } else if (k <= 64) {
            BURST_BLOCK_SELECT_CALL(float, false, 64);
        } else if (k <= 128) {
            BURST_BLOCK_SELECT_CALL(float, false, 128);
        } else if (k <= 256) {
            BURST_BLOCK_SELECT_CALL(float, false, 256);
        } else if (k <= 512) {
            BURST_BLOCK_SELECT_CALL(float, false, 512);
        } else if (k <= 1024) {
            BURST_BLOCK_SELECT_CALL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BURST_BLOCK_SELECT_CALL(float, false, 2048);
#endif
        }
    }
}

void runBurstBlockSelectPair(
        Tensor<float, 3, true>& inK,
        Tensor<int, 3, true>& inV,
        Tensor<float, 3, true>& outK,
        Tensor<int, 3, true>& outV,
        bool dir,
        int k,
        cudaStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 1);
        } else if (k <= 32) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 32);
        } else if (k <= 64) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 64);
        } else if (k <= 128) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 128);
        } else if (k <= 256) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 256);
        } else if (k <= 512) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 512);
        } else if (k <= 1024) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 1);
        } else if (k <= 32) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 32);
        } else if (k <= 64) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 64);
        } else if (k <= 128) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 128);
        } else if (k <= 256) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 256);
        } else if (k <= 512) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 512);
        } else if (k <= 1024) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BURST_BLOCK_SELECT_PAIR_CALL(float, false, 2048);
#endif
        }
    }
}

} // namespace gpu
} // namespace faiss
