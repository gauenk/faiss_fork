/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/BurstBlockSelectKernel.cuh>
#include <faiss/gpu/utils/Limits.cuh>

#define BURST_BLOCK_SELECT_DECL(TYPE, DIR, WARP_Q)                     \
    extern void runBurstBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(     \
            Tensor<TYPE, 3, true>& in,                           \
            Tensor<TYPE, 3, true>& outK,                         \
            Tensor<int, 3, true>& outV,                          \
            bool dir,                                            \
            int k,                                               \
            cudaStream_t stream);                                \
                                                                 \
    extern void runBurstBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_( \
            Tensor<TYPE, 3, true>& inK,                          \
            Tensor<int, 3, true>& inV,                           \
            Tensor<TYPE, 3, true>& outK,                         \
            Tensor<int, 3, true>& outV,                          \
            bool dir,                                            \
            int k,                                               \
            cudaStream_t stream)

#define BURST_BLOCK_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                         \
    void runBurstBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(                          \
            Tensor<TYPE, 3, true>& in,                                         \
            Tensor<TYPE, 3, true>& outK,                                       \
            Tensor<int, 3, true>& outV,                                        \
            bool dir,                                                          \
            int k,                                                             \
            cudaStream_t stream) {                                             \
        FAISS_ASSERT(in.getSize(0) == outK.getSize(0));                        \
        FAISS_ASSERT(in.getSize(0) == outV.getSize(0));                        \
        FAISS_ASSERT(in.getSize(1) == outK.getSize(1));                        \
        FAISS_ASSERT(in.getSize(1) == outV.getSize(1));                        \
        FAISS_ASSERT(outK.getSize(2) == k);                                    \
        FAISS_ASSERT(outV.getSize(2) == k);                                    \
                                                                               \
        auto grid = dim3(in.getSize(0),in.getSize(1));  		       \
                                                                               \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;    \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        FAISS_ASSERT(k <= WARP_Q);                                             \
        FAISS_ASSERT(dir == DIR);                                              \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        burstBlockSelect<TYPE, int, DIR, WARP_Q, THREAD_Q, kBlockSelectNumThreads>  \
                <<<grid, block, 0, stream>>>(in, outK, outV, kInit, vInit, k); \
        CUDA_TEST_ERROR();                                                     \
    }                                                                          \
                                                                               \
    void runBurstBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_(                      \
            Tensor<TYPE, 3, true>& inK,                                        \
            Tensor<int, 3, true>& inV,                                         \
            Tensor<TYPE, 3, true>& outK,                                       \
            Tensor<int, 3, true>& outV,                                        \
            bool dir,                                                          \
            int k,                                                             \
            cudaStream_t stream) {                                             \
        FAISS_ASSERT(inK.isSameSize(inV));                                     \
        FAISS_ASSERT(outK.isSameSize(outV));                                   \
                                                                               \
        auto grid = dim3(inK.getSize(0));                                      \
                                                                               \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;    \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        FAISS_ASSERT(k <= WARP_Q);                                             \
        FAISS_ASSERT(dir == DIR);                                              \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        burstBlockSelectPair<                                                  \
                TYPE,                                                          \
                int,                                                           \
                DIR,                                                           \
                WARP_Q,                                                        \
                THREAD_Q,                                                      \
                kBlockSelectNumThreads><<<grid, block, 0, stream>>>(           \
                inK, inV, outK, outV, kInit, vInit, k);                        \
        CUDA_TEST_ERROR();                                                     \
    }

#define BURST_BLOCK_SELECT_CALL(TYPE, DIR, WARP_Q) \
    runBurstBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(in, outK, outV, dir, k, stream)

#define BURST_BLOCK_SELECT_PAIR_CALL(TYPE, DIR, WARP_Q)    \
    runBurstBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_( \
            inK, inV, outK, outV, dir, k, stream)
