/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Select.cuh>

namespace faiss {
namespace gpu {

template <
        typename K,
        typename IndexType,
        bool Dir,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void burstBlockSelect(
        Tensor<K, 3, true> in,
        Tensor<K, 3, true> outK,
        Tensor<IndexType, 3, true> outV,
        K initK,
        IndexType initV,
        int k) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            K,
            IndexType,
            Dir,
            Comparator<K>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;
    int col = blockIdx.y;

    int i = threadIdx.x;
    K* inStart = in[row][col][i].data(); // get ptr to "float" value

    // Whole warps must participate in the selection
    int limit = utils::roundDown(in.getSize(2), kWarpSize);

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inStart, (IndexType)i); //locally (per-thread) serially append new datum
        inStart += ThreadsPerBlock; // reduces access time by using ptr athrimetic
    }

    // Handle last remainder fraction of a warp of elements
    if (i < in.getSize(2)) {
        heap.addThreadQ(*inStart, (IndexType)i);
    }

    heap.reduce(); // warp reduce using shared memory; quite intensive code: "Merge..."

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        outK[row][col][i] = smemK[i];
        outV[row][col][i] = smemV[i];
    }
}

template <
        typename K,
        typename IndexType,
        bool Dir,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void burstBlockSelectPair(
        Tensor<K, 3, true> inK,
        Tensor<IndexType, 3, true> inV,
        Tensor<K, 3, true> outK,
        Tensor<IndexType, 3, true> outV,
        K initK,
        IndexType initV,
        int k) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            K,
            IndexType,
            Dir,
            Comparator<K>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;
    int col = blockIdx.y;

    int i = threadIdx.x;
    K* inKStart = inK[row][col][i].data();
    IndexType* inVStart = inV[row][col][i].data();

    // Whole warps must participate in the selection
    int limit = utils::roundDown(inK.getSize(2), kWarpSize);

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inKStart, *inVStart);
        inKStart += ThreadsPerBlock;
        inVStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < inK.getSize(2)) {
        heap.addThreadQ(*inKStart, *inVStart);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        outK[row][col][i] = smemK[i];
        outV[row][col][i] = smemV[i];
    }
}

void runBurstBlockSelect(
        Tensor<float, 3, true>& in,
        Tensor<float, 3, true>& outKeys,
        Tensor<int, 3, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

void runBurstBlockSelectPair(
        Tensor<float, 3, true>& inKeys,
        Tensor<int, 3, true>& inIndices,
        Tensor<float, 3, true>& outKeys,
        Tensor<int, 3, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

void runBurstBlockSelect(
        Tensor<half, 3, true>& in,
        Tensor<half, 3, true>& outKeys,
        Tensor<int, 3, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

void runBurstBlockSelectPair(
        Tensor<half, 3, true>& inKeys,
        Tensor<int, 3, true>& inIndices,
        Tensor<half, 3, true>& outKeys,
        Tensor<int, 3, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
