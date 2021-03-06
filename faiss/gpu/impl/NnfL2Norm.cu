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
	int patchsize, int nblocks, TVec TVecMax) {
    extern __shared__ char smemByte[]; // #warps * RowTileSize * ColTileSize * BlockTileSize elements
    float* smem = (float*)smemByte;

    // get the cuda vars 
    IndexType numWarps = utils::divUp(blockDim.x, kWarpSize);
    IndexType laneId = getLaneId();
    // IndexType laneId = threadIdx.x & 0x1f;
    IndexType threadId = threadIdx.x;
    IndexType warpId = threadId / kWarpSize;

    // where do we start the batch within our "thread"
    IndexType height = vals.getSize(0);
    IndexType width = vals.getSize(1);
    IndexType heightPadded = target.getSize(1);
    IndexType widthPadded = target.getSize(2);
    // int heightPadded = 36;
    // int widthPadded = 36;

    // variables needed to use our weird access pattern
    int nftrs = ref.getSize(0);
    int ps = patchsize;
    int psSub1 = patchsize-1;
    int ps2 = patchsize*patchsize;
    IndexType dim = nftrs*ps2;
    float nmlz = 1./(nftrs*ps2);
    int pad = patchsize/2 + nblocks/2;//utils::divDown(patchsize,2);

    // start of thread's batch
    IndexType rowStart = RowTileSize*(blockIdx.x);
    IndexType colStart = ColTileSize*(blockIdx.y);
    IndexType blockStart = BlockTileSize*(blockIdx.z);
    IndexType rowStartPix = rowStart+(nblocks/2);
    IndexType colStartPix = colStart+(nblocks/2);
    

    // determine if our batchsize is too big for the location;
    // the batchsize at compile time might not be a multiple of batchsize at compute time.
    bool lastRowTile = (rowStart + RowTileSize - 1) >= vals.getSize(0);
    bool lastColTile = (colStart + ColTileSize - 1) >= vals.getSize(1);
    bool lastBlockTile = (blockStart + BlockTileSize - 1) >= vals.getSize(2);
    // bool lastTile = lastBlockTile || lastRowTile || lastColTile;
    bool lastTile = lastBlockTile || lastRowTile || lastColTile;

    // add pads since we assume input images have a boarder we don't care about
    // rowStartPaddedImg = rowStart + pad;
    // colStartPaddedImg = colStart + pad;

    // accumulate in f32
    float pixNorm[RowTileSize][ColTileSize][BlockTileSize];

    if (lastTile) { // our batchsizes are too big since we've run out of samples
        // We are handling the very end of the input matrix rows
	IndexType numRowIters = vals.getSize(0) - rowStart;
	numRowIters = inline_min(RowTileSize,(int)numRowIters);
	IndexType numColIters = vals.getSize(1) - colStart;
	numColIters = inline_min(ColTileSize,(int)numColIters);
	IndexType numBlockIters = blocks.getSize(0)-blockStart;
	numBlockIters = inline_min(BlockTileSize,(int)numBlockIters);
	// printf("LAST TILE!\n");

        for (IndexType row = 0; row < numRowIters; ++row) {
	  for (IndexType col = 0; col < numColIters; ++col) {
	    for (IndexType blk = 0; blk < numBlockIters; ++blk) {

	      IndexType tRowStart = blocks[blockStart+blk][0]+rowStartPix+row;
	      IndexType tColStart = blocks[blockStart+blk][1]+colStartPix+col;

	      bool legalRowTgt = legal_access(tRowStart,(psSub1),0,heightPadded);
	      bool legalColTgt = legal_access(tColStart,(psSub1),0,widthPadded);
	      bool legalAccess = legalRowTgt && legalColTgt;
	      
	      // if legal access is false for any patch index,
	      // this thread's work (and the other's on the same patch) are voided.

	      if (NormLoop) {
		pixNorm[0][0][0] = 0;

		for (IndexType txIndex = threadIdx.x;
		     txIndex < dim;
		     txIndex += blockDim.x) {

		  IndexType ftr = txIndex % nftrs;
		  IndexType cmod = txIndex / nftrs;
		  IndexType hIdx = (cmod / patchsize);
		  IndexType wIdx = (cmod % patchsize);
		  
		  IndexType targetRow = tRowStart + hIdx;
		  IndexType targetCol = tColStart + wIdx;
		  IndexType refRow = rowStartPix + row + hIdx;
		  IndexType refCol = colStartPix + col + wIdx;


		  TVec target_val = legalAccess ?
		    target[ftr][targetRow][targetCol]
		    : TVecMax;
		  TVec ref_val = ref[ftr][refRow][refCol];
		  TVec delta_val = Math<TVec>::sub(ref_val,target_val);

		  delta_val = Math<TVec>::mul(delta_val,delta_val);
		  pixNorm[0][0][0] = pixNorm[0][0][0] + Math<TVec>::reduceAdd(delta_val);

		}
	      } else {


		IndexType ftr = threadIdx.x % nftrs;
		IndexType cmod = threadIdx.x / nftrs;
		IndexType hIdx = (cmod / patchsize);
		IndexType wIdx = (cmod % patchsize);

		IndexType targetRow = tRowStart + hIdx;
		IndexType targetCol = tColStart + wIdx;
		IndexType refRow = rowStartPix + row + hIdx;
		IndexType refCol = colStartPix + col + wIdx;
		  

		TVec target_val = legalAccess ?
		  target[ftr][targetRow][targetCol]
		  : TVecMax;
		TVec ref_val = ref[ftr][refRow][refCol];
		TVec delta_val = Math<TVec>::sub(ref_val,target_val);

		delta_val = Math<TVec>::mul(delta_val,delta_val);
		pixNorm[0][0][0] = Math<TVec>::reduceAdd(delta_val);
	      }

	      pixNorm[0][0][0] = warpReduceAllSum(pixNorm[0][0][0]);
	      if (laneId == 0) {
		IndexType smemRowIdx = row;
		IndexType smemColIdx = col * RowTileSize;
		IndexType smemBlockIdx = blk * ColTileSize * RowTileSize;
		IndexType smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		IndexType smemIdx = smemBatchIdx * numWarps + warpId;

		// IndexType smemRowIdx = row;
		// IndexType smemColIdx = col * numRowIters;
		// IndexType smemBlockIdx = blk * numRowIters * numColIters;
		// IndexType smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		// IndexType smemIdx = smemBatchIdx * numWarps + warpId;
		smem[smemIdx]=pixNorm[0][0][0];
		// smem[0]=pixNorm[0][0][0];
	      }
	    }
	  }
	}
    }else {
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

            for (IndexType txIndex = threadIdx.x;
		 txIndex < dim;
                 txIndex += blockDim.x) {
#pragma unroll
	      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
		for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		  for (int blk = 0; blk < BlockTileSize; ++blk) {

		    IndexType tRowStart = blocks[blockStart+blk][0]+rowStartPix+row;
		    IndexType tColStart = blocks[blockStart+blk][1]+colStartPix+col;

		    bool legalRowTgt = legal_access(tRowStart,(psSub1),0,heightPadded);
		    bool legalColTgt = legal_access(tColStart,(psSub1),0,widthPadded);
		    bool legalAccess = legalRowTgt && legalColTgt;

		    IndexType ftr = txIndex % nftrs;
		    IndexType cmod = txIndex / nftrs;
		    IndexType hIdx = (cmod / patchsize);
		    IndexType wIdx = (cmod % patchsize);

		    IndexType targetRow = tRowStart + hIdx;
		    IndexType targetCol = tColStart + wIdx;
		    IndexType refRow = rowStartPix + row + hIdx;
		    IndexType refCol = colStartPix + col + wIdx;


		    TVec target_val = legalAccess ?
		      target[ftr][targetRow][targetCol]
		      : TVecMax;
		    TVec ref_val = ref[ftr][refRow][refCol];
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

		  IndexType tRowStart = blocks[blockStart+blk][0]+rowStartPix+row;
		  IndexType tColStart = blocks[blockStart+blk][1]+colStartPix+col;

		  bool legalRowTgt = legal_access(tRowStart,(psSub1),0,heightPadded);
		  bool legalColTgt = legal_access(tColStart,(psSub1),0,widthPadded);
		  bool legalAccess = legalRowTgt && legalColTgt;

		  IndexType ftr = threadIdx.x % nftrs;
		  IndexType cmod = threadIdx.x / nftrs;
		  IndexType hIdx = (cmod / patchsize);
		  IndexType wIdx = (cmod % patchsize);

		  IndexType targetRow = tRowStart + hIdx;
		  IndexType targetCol = tColStart + wIdx;
		  IndexType refRow = rowStartPix + row + hIdx;
		  IndexType refCol = colStartPix + col + wIdx;


		  TVec target_val = legalAccess ?
		    target[ftr][targetRow][targetCol]
		    : TVecMax;
		  TVec ref_val = ref[ftr][refRow][refCol];
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
		  tmp[row][col][blk] = Math<TVec>::mul(tmp[row][col][blk],
						       tmp[row][col][blk]);
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

        // Sum up all parts within each warp
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
		IndexType smemRowIdx = row;
		IndexType smemColIdx = col * RowTileSize;
		IndexType smemBlockIdx = blk * ColTileSize * RowTileSize;
		IndexType smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		IndexType smemIdx = smemBatchIdx * numWarps + warpId;
		smem[smemIdx] = pixNorm[row][col][blk];
	      }
	    }
	  }
        }
    }

    // printf(" pre sync threads \n");
    __syncthreads();
    // printf(" post sync threads \n");

    // Sum across warps
    // We swap "warpId" with "laneId" so a single Warp (the first one)
    // can read back the elements into the local memory.
    if (warpId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	  for (int blk = 0; blk < BlockTileSize; ++blk) {
	    IndexType smemRowIdx = row;
	    IndexType smemColIdx = col * RowTileSize;
	    IndexType smemBlockIdx = blk * ColTileSize * RowTileSize;
	    IndexType smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
	    IndexType smemIdx = smemBatchIdx * numWarps + laneId;
	    pixNorm[row][col][blk] = laneId < numWarps ? smem[smemIdx] : 0;
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
	      if (lastTile) {
		if (outRow < vals.getSize(0) && outCol < vals.getSize(1) && outBlock < vals.getSize(2)) {
		  vals[outRow][outCol][outBlock] = NormSquared ?
		    ConvertTo<float>::to(pixNorm[row][col][blk])
		    : sqrtf(ConvertTo<float>::to(pixNorm[row][col][blk]));
		}
	      } else {
		vals[outRow][outCol][outBlock] = NormSquared
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
	int patchsize, int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    IndexType maxThreads = (IndexType)getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 1;
    constexpr int colTileSize = 1;
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
		  true><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, patchsize, nblocks, TVecMax); \
            } else {                                                          \
                nnfl2NormRowMajor<                                               \
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        true,                                                 \
		  false><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, patchsize, nblocks, TVecMax); \
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
		true><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, patchsize, nblocks, TVecMax); \
            } else {                                                          \
                nnfl2NormRowMajor<                                               \
                        TYPE_T,                                               \
                        TYPE_TVEC,                                            \
                        IndexType,                                            \
                        rowTileSize,                                          \
                        colTileSize,                                          \
                        blockTileSize,                                          \
                        false,                                                \
		  false><<<grid, block, smem, stream>>>(REF, TARGET, blocks, vals, patchsize, nblocks, TVecMax); \
            }                                                                 \
        }                                                                     \
    } while (0)

    // std::cout << "sizeof " << sizeof(TVec) << ", "<< sizeof(T) << std::endl;
    // auto ref_can_recast = ref.template canCastResize<TVec>();
    // auto target_can_recast = target.template canCastResize<TVec>();

    auto ref_can_recast = false;//ref.template canCastResize<TVec>();
    auto target_can_recast = false;//target.template canCastResize<TVec>();

    // compute numThreads
    int nftrs = ref.getSize(0);
    IndexType dim = patchsize*patchsize*nftrs;
    bool normLoop = dim > maxThreads;
    IndexType numThreads = std::min(dim, maxThreads);
    // numThreads = utils::roundUp(numThreads,kWarpSize); // round-up for warp reduce.

    // compute number of Warps per kernel with size "numThreads"
    IndexType nWarps = utils::divUp(numThreads, kWarpSize);

    // compute number of Grids
    int height = vals.getSize(0);
    int width = vals.getSize(1);
    int blockBatchSize = blocks.getSize(0);
    int numToComp = height * width * blockBatchSize;
    int numToCompPerKernel = rowTileSize * colTileSize * blockTileSize;
    int numHeightBlocks = utils::divUp(height, rowTileSize);
    int numWidthBlocks = utils::divUp(width, colTileSize);
    int numBlockBlocks = utils::divUp(blockBatchSize, blockTileSize);
    int nBlocks = utils::divUp(numToComp,numToCompPerKernel);
    // std::cout << "numToComp " << numToComp << std::endl;
    // std::cout << "numToCompPerKernel " << numToCompPerKernel << std::endl;
    // std::cout << "nBlocks " << nBlocks << std::endl;
    // std::cout << "blockBatchSize " << blockBatchSize << std::endl;
    // std::cout << "normLoop: " << normLoop << std::endl;
    // std::cout << "numWarps: " << nWarps << std::endl;
    // std::cout << "NumThreads: " << numThreads << std::endl;

    // std::cout << "height " << height << std::endl;
    // std::cout << "width " << width << std::endl;

    // std::cout << "heightPadded " << target.getSize(1) << std::endl;
    // std::cout << "widthPadded " << target.getSize(2) << std::endl;

    // std::cout << "numHeightBlocks " << numHeightBlocks << std::endl;
    // std::cout << "numWidthBlocks " << numWidthBlocks << std::endl;
    // std::cout << "numBlockBlocks " << numBlockBlocks << std::endl;

    if (ref_can_recast && target_can_recast) {
        std::cout << "recasting!"  << std::endl;
        // Can load using the vectorized type
        auto refV = ref.template castResize<TVec>();
        auto targetV = target.template castResize<TVec>();
	// printf("ref.stride: (%ld,%ld,%ld)\n",
	//        ref.getStride(0),ref.getStride(1),ref.getStride(2));
	// printf("target.stride: (%ld,%ld,%ld)\n",
	//        target.getStride(0),target.getStride(1),target.getStride(2));
	// printf("refV.shape: (%ld,%ld,%ld)\n",
	//        refV.getSize(0),refV.getSize(1),refV.getSize(2));
	// printf("targetV.shape: (%ld,%ld,%ld)\n",
	//        targetV.getSize(0),targetV.getSize(1),targetV.getSize(2));
	// printf("refV.stride: (%ld,%ld,%ld)\n",
	//        refV.getStride(0),refV.getStride(1),refV.getStride(2));
	// printf("targetV.stride: (%ld,%ld,%ld)\n",
	//        targetV.getStride(0),targetV.getStride(1),targetV.getStride(2));

        // auto grid = dim3(nBlocks);
        auto grid = dim3(numHeightBlocks,numWidthBlocks,numBlockBlocks);
        auto block = dim3(numThreads);

        auto smem = sizeof(float) * numToCompPerKernel * nWarps;

	// weird convserion ... idk
	TVec* tmpTVec;
	float tmp[1];
	tmp[0] = 100.;
	tmpTVec = reinterpret_cast<TVec*>(tmp);//Limits<TVec>::getMax();
	TVec TVecMax = tmpTVec[0];

        RUN_NNF_L2_ROW_MAJOR(T, TVec, refV, targetV);
    } else {
        // Can't load using the vectorized type

        // auto grid = dim3(nBlocks);
        auto grid = dim3(numHeightBlocks,numWidthBlocks,numBlockBlocks);
        auto block = dim3(numThreads);

        auto smem = sizeof(float) * numToCompPerKernel * nWarps;


	// weird convserion ... idk
	float* tmpTVec;
	float tmp[1];
	tmp[0] = 100.;
	tmpTVec = reinterpret_cast<float*>(tmp);//Limits<TVec>::getMax();
	float TVecMax = tmpTVec[0];

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
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    if (ref.canUseIndexType<int>()) {
        runNnfL2Norm<float, float4, int>(
		ref, target, blocks, vals, patchsize, nblocks, normSquared, stream);
    } else {
        auto refCast = ref.castIndexType<long>();
        auto targetCast = target.castIndexType<long>();
        auto blocksCast = blocks.castIndexType<long>();
        auto valCast = vals.castIndexType<long>();

        runNnfL2Norm<float, float4, long>(
		refCast, targetCast, blocksCast, valCast, patchsize, nblocks, normSquared, stream);
    }
}

void runNnfL2Norm(
        Tensor<half, 3, true>& ref,
        Tensor<half, 3, true>& target,
        Tensor<int, 2, true>& blocks,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream) {
    if (ref.canUseIndexType<int>()) {
        runNnfL2Norm<half, half2, int>(
		ref, target, blocks, vals, patchsize, nblocks, normSquared, stream);
    } else {
        auto refCast = ref.castIndexType<long>();
        auto targetCast = target.castIndexType<long>();
        auto blocksCast = blocks.castIndexType<long>();
        auto valCast = vals.castIndexType<long>();

        runNnfL2Norm<half, half2, long>(
		refCast, targetCast, blocksCast, valCast, patchsize, nblocks, normSquared, stream);
    }
}

} // namespace gpu
} // namespace faiss
