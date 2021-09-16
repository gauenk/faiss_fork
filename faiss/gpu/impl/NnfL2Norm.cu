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
template <
        typename T,
        typename TVec,
        typename IndexType,
        int RowTileSize,
        int ColTileSize,
        int BlockTileSize,
        bool NormLoop,
        bool NormSquared>
__global__ void nnfl2NormRowMajor(
        Tensor<TVec, 3, true, IndexType> ref,
        Tensor<TVec, 3, true, IndexType> target,
        Tensor<int, 2, true, IndexType> blocks,
        Tensor<float, 3, true, IndexType> vals,
        Tensor<int, 4, true, IndexType> locs,
	int patchsize, int nblocks) {
    extern __shared__ char smemByte[]; // #warps * RowTileSize * ColTileSize * BlockTileSize elements
    float* smem = (float*)smemByte;

    IndexType numWarps = utils::divUp(blockDim.x*blockDim.y*blockDim.z, kWarpSize);
    IndexType threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    IndexType tidx = threadId;
    IndexType warpId = threadId / kWarpSize;
    IndexType laneId = warpId % kWarpSize;


    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    bool lastColTile = (blockIdx.y == (gridDim.y - 1));
    bool lastBlockTile = (blockIdx.z == (gridDim.z - 1));
    // This is the "blockIdx" gives us enough of these so this should be okay.
    IndexType rowStart = RowTileSize * blockIdx.x;
    IndexType colStart = ColTileSize * blockIdx.y;
    IndexType blockStart = BlockTileSize * blockIdx.z;

    int counter = 0; // counter to specify index within (patchsize)^2 * nftrs
    int nftrs = ref.getSize(0);
    int ps2 = patchsize*patchsize;
    int pad = utils::divDown(ps2,2);
    IndexType blockIndex = threadId % blocks.getSize(0);
    IndexType ftrIndex = utils::divDown(threadId,blocks.getSize(0));

    // accumulate in f32
    float pixNorm[RowTileSize][ColTileSize][BlockTileSize];

    if (lastRowTile || lastColTile || lastBlockTile) {
        // We are handling the very end of the input matrix rows
        for (IndexType row = 0; row < ref.getSize(1) - rowStart; ++row) {
	  for (IndexType col = 0; col < ref.getSize(2) - colStart; ++col) {
	    for (IndexType blk = 0; col < blocks.getSize(0) - blockStart; ++blk) {
	      if (NormLoop) {
		IndexType tRowStart = blocks[blockStart+blk][0]-pad+rowStart;
		IndexType tColStart = blocks[blockStart+blk][1]-pad+colStart;
		pixNorm[0][0][0] = 0;

		for (IndexType txIndex = threadIdx.x;
		     txIndex < ref.getSize(0);
		     txIndex += blockDim.x) {

		  IndexType cmod = counter % ps2;
		  IndexType hIdx = cmod / patchsize;
		  IndexType wIdx = cmod % patchsize;
		  IndexType ftr = counter / ps2;
		  IndexType targetRow = tRowStart + row + hIdx;
		  IndexType targetCol = tColStart + col + wIdx;

		  TVec target_val = target[ftr][targetRow][targetCol];
		  TVec ref_val = ref[ftr][rowStart + row][colStart + col];
		  TVec delta_val = Math<TVec>::sub(ref_val,target_val);

		  delta_val = Math<TVec>::mul(delta_val,delta_val);
		  pixNorm[0][0][0] = pixNorm[0][0][0] + Math<TVec>::reduceAdd(delta_val);

		  counter += 1;
		}
	      } else {
		TVec ref_val = ref[tidx][rowStart + row][colStart + col];
		TVec target_val = target[tidx][tRowStart + row][tColStart + col];
		TVec delta_val = Math<TVec>::sub(ref_val,target_val);
		delta_val = Math<TVec>::mul(delta_val,delta_val);
		pixNorm[0][0][0] = Math<TVec>::reduceAdd(delta_val);
	      }

	      pixNorm[0][0][0] = warpReduceAllSum(pixNorm[0][0][0]);
	      if (laneId == 0) {
		smem[0]=pixNorm[0];
		//smem[row * ColTileSize * numWarps + col * numWarps + warpId]=pixNorm[0];
	      }
	    }
	  }
	}
    } else {
        // We are guaranteed that all RowTileSize rows are available in
        // [rowStart, rowStart + RowTileSize)

        if (NormLoop) {
            // A single block of threads is not big enough to span each
            // vector
  	    TVec tmp[RowTileSize][ColTileSize][BlockTileSize];

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {
		  pixNorm[row][col][blk] = 0;
		}
	      }
	    }

            for (IndexType stratX = threadIdx.x; stratX < ps2*nftrs;
                 stratX += blockDim.x) {
#pragma unroll
	      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
		for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		  for (int blk = 0; blk < BlockTileSize; ++blk) {
		    IndexType ftr = stratX % nftrs;
		    IndexType tIdxH = utils::divUp(stratX,nftrs);

		    IndexType tRowStart = blocks[blockStart+blk][0]-pad+rowStart;
		    IndexType tColStart = blocks[blockStart+blk][1]-pad+colStart;
		    TVec ref_val = ref[ftr][rowStart + row][colStart + col];
		    TVec target_val = target[ftr][tRowStart + row][tColStart + col];
                    tmp[row][col][blk] = Math<TVec>::sub(ref_val,target_val);
		  }
		}
	      }

#pragma unroll
	      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
		for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		  for (int blk = 0; blk < BlockTileSize; ++blk) {
		    tmp[row][col][blk] = Math<TVec>::mul(tmp[row][col][blk],tmp[row][col][blk]);
		  }
		}
	      }

#pragma unroll
	      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
		for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		  for (int blk = 0; blk < BlockTileSize; ++blk) {
		    pixNorm[row][col][blk] =
		      pixNorm[row][col][blk] + Math<TVec>::reduceAdd(tmp[row][col][blk]);
		  }
		}
	      }
	    }
        } else {
            TVec tmp[RowTileSize][ColTileSize][BlockTileSize];

            // A block of threads is the exact size of the vector
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {
		    TVec ref_val = ref[ftr][rowStart + row][colStart + col];
		    TVec target_val = target[ftr][tRowStart + row][tColStart + col];
		    tmp[row][col][blk] = Math<TVec>::sub(ref_val,target_val);
		}
	      }
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {
		  tmp[row][col][blk] = Math<TVec>::mul(tmp[row][col][blk], tmp[row][col][blk]);
		}
	      }
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {
		  pixNorm[row][col][blk] = Math<TVec>::reduceAdd(tmp[row][col][blk]);
		}
	      }
	    }
	}

        // Sum up all parts in each warp
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	  for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	    for (int blk = 0; blk < BlockTileSize; ++blk) {
	      pixNorm[row][col][blk] = warpReduceAllSum(pixNorm[row][col][blk]);
	    }
	  }
	}

        if (laneId == 0) {
#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int blk = 0; blk < BlockTileSize; ++blk) {
		smem[0] = pixNorm[row][col][blk];
		//smem[row * ColTileSize * numWarps + col * numWarps + warpId] = pixNorm[row*ColTileSize+col];
	      }
	    }
	  }
        }
    }

    __syncthreads();

    // Sum across warps
    if (warpId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	  for (int blk = 0; blk < BlockTileSize; ++blk) {
	    pixNorm[row][col][blk] = laneId < numWarps ? smem[0] : 0;
	    // pixNorm[row*ColTileSize+col] =
	    //   laneId < numWarps ? smem[row * ColTileSize * numWarps + col * numWarps + warpId] : 0;
	  }
        }
      }

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	  for (int blk = 0; blk < BlockTileSize; ++blk) {
	    pixNorm[row][col][blk] = warpReduceAllSum(pixNorm[row][col][blk]);
	  }
        }
      }

      // Write out answer
      if (laneId == 0) {
#pragma unroll
	for (int row = 0; row < RowTileSize; ++row) {
	  int outRow = rowStart + row;
#pragma unroll
	  for (int col = 0; col < ColTileSize; ++col) {
	    int outCol = colStart + col;
#pragma unroll
	    for (int blk = 0; blk < BlockTileSize; ++blk) {
	      int outBlock = blockStart + blk;
	      if (lastRowTile || lastColTile || lastBlockTile) {
		if (outRow < vals.getSize(0) && outCol < vals.getSize(1)) {
		  vals[outRow][outCol][outBlock] = NormSquared ?
		    ConvertTo<float>::to(pixNorm[row][col][blk])
		    : sqrtf(ConvertTo<float>::to(pixNorm[row][col][blk]));
		}
	      } else {
		vals[outRow][outCol][0] = NormSquared
		  ? ConvertTo<float>::to(pixNorm[row][col][blk])
		  : sqrtf(ConvertTo<float>::to(pixNorm[row][col][blk]));
	      }
	    }
	  }
	}
      }
    }
}

template <typename T, typename TVec, typename IndexType>
void runNnfL2Norm(
	Tensor<T, 3, true, IndexType>& ref,
	Tensor<T, 3, true, IndexType>& target,
        Tensor<int, 2, true, IndexType>& blocks,
        Tensor<float, 3, true, IndexType>& vals,
        Tensor<int, 4, true, IndexType>& locs,
	int patchsize, int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    IndexType maxThreads = (IndexType)getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 4;
    constexpr int colTileSize = 4;
    constexpr int blockTileSize = 4;

#define RUN_NNF_L2_ROW_MAJOR(TYPE_T, TYPE_TVEC, REF, TARGET)			\
    do {                                                                      \
        if (normLoop) {                                                       \
            if (normSquared) {                                                \
                nnfl2NormRowMajor<                                               \
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        true,                                                 \
		  true><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, locs, patchsize, nblocks); \
            } else {                                                          \
                nnfl2NormRowMajor<                                               \
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        true,                                                 \
		  false><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, locs, patchsize, nblocks); \
            }                                                                 \
        } else {                                                              \
            if (normSquared) {                                                \
	      nnfl2NormRowMajor<						\
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        false,                                                \
		true><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, locs, patchsize, nblocks); \
            } else {                                                          \
                nnfl2NormRowMajor<                                               \
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        false,                                                \
		  false><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, locs, patchsize, nblocks); \
            }                                                                 \
        }                                                                     \
    } while (0)

    auto ref_can_recast = ref.template canCastResize<TVec>();
    auto target_can_recast = target.template canCastResize<TVec>();

    // compute numThreads
    int nftrs = ref.getSize(0);
    int numBlocksBatch = blocks.getSize(0);
    int numBlocksX = (int)std::floor(std::sqrt(numBlocksBatch*1.0));
    int numBlocksY = utils::divUp(numBlocksBatch,numBlocksX);
    int dim = patchsize*patchsize*nftrs;
    // auto dim = patchsize*patchsize;
    int cubeRootMaxThreads = (int)utils::pow(maxThreads*1.0,.33);
    bool normLoop = dim > cubeRootMaxThreads;
    int numThreads = std::min(dim, cubeRootMaxThreads);
    int nWarps = utils::divUp(numThreads, kWarpSize);

    if (ref_can_recast && target_can_recast) {
        // Can load using the vectorized type
        auto refV = ref.template castResize<TVec>();
        auto targetV = target.template castResize<TVec>();

	auto gridx = utils::divUp(refV.getSize(1), rowTileSize);
	auto gridy = utils::divUp(refV.getSize(2), colTileSize);
	auto gridz = utils::divUp(blocks.getSize(0), blockTileSize);

        auto grid = dim3(gridx,gridy,gridz);
        auto block = dim3(numThreads,numThreads,numThreads);

        auto smem = sizeof(float) * rowTileSize * colTileSize * blockTileSize * nWarps;

        RUN_NNF_L2_ROW_MAJOR(T, TVec, refV, targetV);
    } else {
        // Can't load using the vectorized type

	auto gridx = utils::divUp(refV.getSize(1), rowTileSize);
	auto gridy = utils::divUp(refV.getSize(2), colTileSize);
	auto gridz = utils::divUp(blocks.getSize(0), blockTileSize);

        auto grid = dim3(gridx,gridy,gridz);
        auto block = dim3(numThreads,numThreads,numThreads);

        auto smem = sizeof(float) * rowTileSize * colTileSize * blockTileSize * nWarps;

        RUN_NNF_L2_ROW_MAJOR(T, T, ref, target);
    }

#undef RUN_NNF_L2

    CUDA_TEST_ERROR();
}

void runNnfL2Norm(
        Tensor<float, 3, true>& ref,
        Tensor<float, 3, true>& target,
        Tensor<int, 2, true>& blocks,
        Tensor<float, 3, true>& vals,
        Tensor<int, 4, true>& locs,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    if (ref.canUseIndexType<int>()) {
        runNnfL2Norm<float, float4, int>(
		ref, target, blocks, vals, locs, patchsize, nblocks, normSquared, stream);
    } else {
        auto refCast = ref.castIndexType<long>();
        auto targetCast = target.castIndexType<long>();
        auto blocksCast = blocks.castIndexType<long>();
        auto valCast = vals.castIndexType<long>();
        auto locCast = locs.castIndexType<long>();

        runNnfL2Norm<float, float4, long>(
		refCast, targetCast, blocksCast, valCast, locCast, patchsize, nblocks, normSquared, stream);
    }
}

void runNnfL2Norm(
        Tensor<half, 3, true>& ref,
        Tensor<half, 3, true>& target,
        Tensor<int, 2, true>& blocks,
        Tensor<float, 3, true>& vals,
        Tensor<int, 4, true>& locs,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    if (ref.canUseIndexType<int>()) {
        runNnfL2Norm<half, half2, int>(
		ref, target, blocks, vals, locs, patchsize, nblocks, normSquared, stream);
    } else {
        auto refCast = ref.castIndexType<long>();
        auto targetCast = target.castIndexType<long>();
        auto blocksCast = blocks.castIndexType<long>();
        auto valCast = vals.castIndexType<long>();
        auto locCast = locs.castIndexType<long>();

        runNnfL2Norm<half, half2, long>(
		refCast, targetCast, blocksCast, valCast, locCast, patchsize, nblocks, normSquared, stream);
    }
}

} // namespace gpu
} // namespace faiss
