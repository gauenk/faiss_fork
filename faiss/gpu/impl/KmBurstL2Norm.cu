/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/NnfL2Norm.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

#define inline_min(a, b) ( (a) < (b) ? (a) : (b) )
// #define macro_max(a, b) (a > b)? a: b 
#define legal_access(a, b, c, d) ((((a) >= (c)) && (((a)+(b)) < (d))) ? true : false)

// inline_min(float x, float y);
// template<type T> inline T
// inline_min(T x, T y) { return x < y ? x : y; }


// Input: (batch x dim)
// Output: (batch norm)
// Done under the presumption that the dimension size is not too large
// (<10k or so), since there wouldn't be enough parallelism applying a
// single block to the problem. Also that each vector is large enough
// (>64), since a single block works on multiple rows' norms at the
// same time.
// T: the type we are doing the math in (e.g., float, half)
// TVec: the potentially vectorized type we are loading in (e.g.,
// float4, half2)
template <typename T,
	  int RowTileSize,
	  int ColTileSize,
	  int BlockTileSize,
	  bool NormLoop,
	  bool NormSquared>
__global__ void runKmBurstL2NormKernel(
	Tensor<T, 5, true, int> burst,
	Tensor<T, 4, true, int> ave,
        Tensor<int, 5, true, int> blocks,
        Tensor<float, 3, true, int> vals,
	int patchsize, int nblocks, T TVecMax) {
  // #warps * RowTileSize * ColTileSize * BlockTileSize elements
  extern __shared__ char smemByte[]; 

  
    
}

template <typename T>
void runKmBurstL2Norm(
	Tensor<T, 5, true, int>& burst,
	Tensor<T, 4, true, int>& ave,
        Tensor<int, 5, true, int>& blocks,
        Tensor<float, 3, true, int>& vals,
	int patchsize, int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    int maxThreads = (int)getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 2;
    constexpr int colTileSize = 2;
    constexpr int blockTileSize = 4;

#define RUN_KMB_L2_ROW_MAJOR(TYPE_T, BURST)			\
    do {                                                                      \
        if (normLoop) {                                                       \
            if (normSquared) {                                                \
                runKmBurstL2NormKernel<                                               \
                        TYPE_T,                                               \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        true,                                                 \
			  true><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
            } else {                                                          \
                runKmBurstL2NormKernel<                                               \
                        TYPE_T,                                               \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        true,                                                 \
		  false><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
            }                                                                 \
        } else {                                                              \
            if (normSquared) {                                                \
	      runKmBurstL2NormKernel<						\
                        TYPE_T,                                               \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        false,                                                \
		true><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
            } else {                                                          \
                runKmBurstL2NormKernel<                                               \
                        TYPE_T,                                               \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        false,                                                \
		  false><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
            }                                                                 \
        }                                                                     \
    } while (0)


    // compute numThreads
    int nframes = burst.getSize(0);
    int nftrs = burst.getSize(1);
    int dim = patchsize*patchsize*nftrs*nframes;
    bool normLoop = dim > maxThreads;
    int numThreads = std::min(dim, maxThreads);
    int nWarps = utils::divUp(numThreads, kWarpSize);
    // numThreads = utils::roundUp(numThreads,kWarpSize); // round-up for warp reduce.
    FAISS_ASSERT(vals.getSize(2) >= blocks.getSize(1));

    // compute number of Grids
    int height = vals.getSize(0);
    int width = vals.getSize(1);
    int blockBatchSize = blocks.getSize(1);
    int numToComp = height * width * blockBatchSize;
    int numToCompPerKernel = rowTileSize * colTileSize * blockTileSize;
    int numHeightBlocks = utils::divUp(height, rowTileSize);
    int numWidthBlocks = utils::divUp(width, colTileSize);
    int numBlockBlocks = utils::divUp(blockBatchSize, blockTileSize);
    int nBlocks = utils::divUp(numToComp,numToCompPerKernel);

    // get grids and threads 
    auto grid = dim3(numHeightBlocks,numWidthBlocks,numBlockBlocks);
    auto block = dim3(numThreads);
    auto smem = sizeof(float) * numToCompPerKernel * nWarps;

    // weird convserion for me... ... idk
    float* tmpTVec;
    float tmp[1];
    tmp[0] = 100.;
    tmpTVec = reinterpret_cast<float*>(tmp);
    float TVecMax = tmpTVec[0];
    RUN_KMB_L2_ROW_MAJOR(T, burst);

#undef RUN_NNF_L2
    CUDA_TEST_ERROR();
}

void runKmBurstL2Norm(
	Tensor<float, 5, true>& burst,
	Tensor<float, 4, true>& ave,
        Tensor<int, 5, true>& blocks,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
  runKmBurstL2Norm<float>(burst, ave, blocks, vals,
			  patchsize, nblocks, normSquared, stream);
}

void runKmBurstL2Norm(
	Tensor<half, 5, true>& burst,
	Tensor<half, 4, true>& ave,
        Tensor<int, 5, true>& blocks,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
  runKmBurstL2Norm<half>(burst, ave, blocks, vals,
			 patchsize, nblocks, normSquared, stream);
}

} // namespace gpu
} // namespace faiss
