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
#define legal_access(a, b, c, d) ((((a) >= (c)) && (((a)+(b)) < (d))) ? true : false)

    __forceinline__ __device__ int hw_boundary(int hw, int max){
      hw = (hw) < 0 ? -(hw) : (hw);
      hw = (hw) > (max) ? (2*(max) - (hw)) : (hw);
      return hw;
    }


    template <typename T,
      int RowTileSize,
      int ColTileSize,
      int BlockTileSize,
      bool NormLoop,
      bool NormSquared>
    __global__ void runKmBurstL2NormKernel(Tensor<T, 5, true, int> centroids,
					   Tensor<T, 4, true, int> ave,
					   Tensor<int, 5, true, int> indices,
					   Tensor<float, 3, true, int> vals,
					   int patchsize, int nblocks, T TVecMax) {

      // #warps * RowTileSize * ColTileSize * BlockTileSize elements
      extern __shared__ char smemByte[];
      float* smem = (float*)smemByte;

      // get the cuda vars 
      int numWarps = utils::divUp(blockDim.x, kWarpSize); 
      int laneId = getLaneId();
      int threadId = threadIdx.x;
      int warpId = threadId / kWarpSize;

      // where do we start the batch within our "thread"
      // int nframes = vals.getSize(0);
      int nftrs = centroids.getSize(0);
      int nframes = centroids.getSize(1);
      int height = centroids.getSize(2);
      int width = centroids.getSize(3);
      int bBatch = vals.getSize(0);
      int hBatch = vals.getSize(1);
      int wBatch = vals.getSize(2);

      // variables needed to use our weird access pattern
      int ps = patchsize;
      int psSub1 = patchsize-1;
      int ps2 = patchsize*patchsize;
      int dim = nftrs*ps2*nframes;
      // float nmlz = 1./(nftrs*ps2*nframes*1.0);
      float inv_frames = 1./(nframes*1.0);
      T zero = ConvertTo<T>::to(0);
      T TMax = 1000;

      // start of thread's batch
      int rowStart = RowTileSize*(blockIdx.x);
      int colStart = ColTileSize*(blockIdx.y);
      int blockStart = BlockTileSize*(blockIdx.z);
      // int nbHalf = nblocks/2;

      // determine if our batchsize is too big for the location;
      // the batchsize at compile time might not be a multiple
      // of batchsize at compute time.
      bool lastRowTile = (rowStart + RowTileSize - 1) >= hBatch;
      bool lastColTile = (colStart + ColTileSize - 1) >= wBatch;
      bool lastBlockTile = (blockStart + BlockTileSize - 1) >= bBatch;
      // bool lastTile = lastBlockTile || lastRowTile || lastColTile;
      bool lastTile = lastBlockTile || lastRowTile || lastColTile;
      int hIdx_b,wIdx_b;
      int psHalf = ps/2;

      // accumulate in f32
      float pixNorm[RowTileSize][ColTileSize][BlockTileSize];

      if (lastTile) { // our batchsizes are too big since we've run out of samples
        // We are handling the very end of the input matrix rows
	int numRowIters = vals.getSize(1) - rowStart;
	numRowIters = inline_min(RowTileSize,(int)numRowIters);
	int numColIters = vals.getSize(2) - colStart;
	numColIters = inline_min(ColTileSize,(int)numColIters);
	int numBlockIters = vals.getSize(0) - blockStart;
	numBlockIters = inline_min(BlockTileSize,(int)numBlockIters);
	// printf("LAST TILE!\n");

        for (int row = 0; row < numRowIters; ++row) {
	  for (int col = 0; col < numColIters; ++col) {
	    for (int blk = 0; blk < numBlockIters; ++blk) {

	      
	      // if legal access is false for any patch index,
	      // this thread's work (and the other's on the same patch) are voided.

	      if (NormLoop) {
		pixNorm[0][0][0] = 0;

		for (int txIndex = threadIdx.x;
		     txIndex < dim;
		     txIndex += blockDim.x) {

		  // features 
		  int ftr = txIndex % nftrs;
		  int fDiv = txIndex / nftrs;

		  // width
		  int wIdx = fDiv % patchsize;
		  int wDiv = fDiv / patchsize;

		  // height
		  int hIdx = wDiv % patchsize;
		  int hDiv = wDiv / patchsize;

		  // frame index
		  int fIdx = hDiv % nframes;

		  // ref image indices
		  int refBlk = blockStart + blk;
		  int refRow = rowStart + row + hIdx - psHalf;
		  int refCol = colStart + col + wIdx - psHalf;

		  // blocks
		  int blRow = indices[0][fIdx][refBlk][refRow][refCol];
		  int blCol = indices[1][fIdx][refBlk][refRow][refCol];

		  // legal accesses only
		  hIdx_b = hw_boundary(blRow,height-1);
		  wIdx_b = hw_boundary(blCol,width-1);
		  refRow = hw_boundary(refRow,hBatch-1); //?
		  refCol = hw_boundary(refCol,wBatch-1); //?

		  // image values
		  T centroids_val = centroids[ftr][fIdx][refBlk][refRow][refCol];
		  T ave_val = ave[ftr][refBlk][refRow][refCol];

		  // compute deltas 
		  T delta_val = Math<T>::sub(centroids_val,ave_val);
		  delta_val = Math<T>::mul(delta_val,delta_val);
		  delta_val = Math<T>::mul(delta_val,inv_frames);
		  pixNorm[0][0][0] = pixNorm[0][0][0] + Math<T>::reduceAdd(delta_val);

		}
	      } else {

		// features 
		int ftr = threadIdx.x % nftrs;
		int fDiv = threadIdx.x / nftrs;

		// width
		int wIdx = fDiv % patchsize;
		int wDiv = fDiv / patchsize;

		// height
		int hIdx = wDiv % patchsize;
		int hDiv = wDiv / patchsize;

		// frame index
		int fIdx = hDiv % nframes;

		// ref image indices
		int refBlk = blockStart + blk;
		int refRow = rowStart + row + hIdx - psHalf;
		int refCol = colStart + col + wIdx - psHalf;

		// blocks
		int blRow = indices[0][fIdx][refBlk][refRow][refCol];
		int blCol = indices[1][fIdx][refBlk][refRow][refCol];

		// legal accesses only
		hIdx_b = hw_boundary(blRow,height-1);
		wIdx_b = hw_boundary(blCol,width-1);
		refRow = hw_boundary(refRow,hBatch-1); //?
		refCol = hw_boundary(refCol,wBatch-1); //?

		// image values
		T centroids_val = centroids[ftr][fIdx][refBlk][refRow][refCol];
		T ave_val = ave[ftr][refBlk][refRow][refCol];

		// compute deltas 
		T delta_val = Math<T>::sub(centroids_val,ave_val);
		delta_val = Math<T>::mul(delta_val,delta_val);
		delta_val = Math<T>::mul(delta_val,inv_frames);
		pixNorm[0][0][0] = Math<T>::reduceAdd(delta_val);
	      }

	      pixNorm[0][0][0] = warpReduceAllSum(pixNorm[0][0][0]);
	      if (laneId == 0) {
		int smemRowIdx = row;
		int smemColIdx = col * RowTileSize;
		int smemBlockIdx = blk * ColTileSize * RowTileSize;
		int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		int smemIdx = smemBatchIdx * numWarps + warpId;
		smem[smemIdx]=pixNorm[0][0][0];
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
	  T tmp[RowTileSize][ColTileSize][BlockTileSize];

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

	  for (int txIndex = threadIdx.x;
	       txIndex < dim;
	       txIndex += blockDim.x) {
#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {

		  // features 
		  int ftr = txIndex % nftrs;
		  int fDiv = txIndex / nftrs;
    		    
		  // width
		  int wIdx = fDiv % patchsize;
		  int wDiv = fDiv / patchsize;
    		    
		  // height
		  int hIdx = wDiv % patchsize;
		  int hDiv = wDiv / patchsize;
    		    
		  // frame index
		  int fIdx = hDiv % nframes;
    		    
		  // ref image indices
		  int refBlk = blockStart + blk;
		  int refRow = rowStart + row + hIdx - psHalf;
		  int refCol = colStart + col + wIdx - psHalf;

		  // blocks
		  int blRow = indices[0][fIdx][refBlk][refRow][refCol];
		  int blCol = indices[1][fIdx][refBlk][refRow][refCol];
    		    
		  // legal accesses only
		  hIdx_b = hw_boundary(blRow,height-1);
		  wIdx_b = hw_boundary(blCol,width-1);
		  refRow = hw_boundary(refRow,hBatch-1); //?
		  refCol = hw_boundary(refCol,wBatch-1); //?

		  // image values
		  T centroids_val = centroids[ftr][fIdx][refBlk][refRow][refCol];
		  T ave_val = ave[ftr][refBlk][refRow][refCol];

		  // compute sub 
		  tmp[row][col][blk] = Math<T>::sub(centroids_val,ave_val);

		}
	      }
	    }


#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int blk = 0; blk < BlockTileSize; ++blk) {
		  tmp[row][col][blk] = Math<T>::mul(tmp[row][col][blk],
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
		  tmp[row][col][blk] = Math<T>::mul(tmp[row][col][blk],inv_frames);
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
		    pixNorm[row][col][blk] + Math<T>::reduceAdd(tmp[row][col][blk]);
		}
	      }
	    }
	  }
        } else {

	  // A block of threads is the exact size of the vector
#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int blk = 0; blk < BlockTileSize; ++blk) {

		// features 
		int ftr = threadIdx.x % nftrs;
		int fDiv = threadIdx.x / nftrs;
    		  
		// width
		int wIdx = fDiv % patchsize;
		int wDiv = fDiv / patchsize;
    		  
		// height
		int hIdx = wDiv % patchsize;
		int hDiv = wDiv / patchsize;
    		  
		// frame index
		int fIdx = hDiv % nframes;

		// ref image indices
		int refBlk = blockStart + blk;
		int refRow = rowStart + row + hIdx - psHalf;
		int refCol = colStart + col + wIdx - psHalf;

		// blocks
		int blRow = indices[0][fIdx][refBlk][refRow][refCol];
		int blCol = indices[1][fIdx][refBlk][refRow][refCol];
    		  
		// legal accesses only
		hIdx_b = hw_boundary(blRow,height-1);
		wIdx_b = hw_boundary(blCol,width-1);
		refRow = hw_boundary(refRow,hBatch-1); //?
		refCol = hw_boundary(refCol,wBatch-1); //?
    		  
		// image values
		T centroids_val = centroids[ftr][fIdx][refBlk][refRow][refCol];
		T ave_val = ave[ftr][refBlk][refRow][refCol];
		pixNorm[row][col][blk] = Math<T>::sub(centroids_val,ave_val);

	      }
	    }
	  }

#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int blk = 0; blk < BlockTileSize; ++blk) {
		pixNorm[row][col][blk] = Math<T>::mul(pixNorm[row][col][blk],
						      pixNorm[row][col][blk]);
	      }
	    }
	  }


#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int blk = 0; blk < BlockTileSize; ++blk) {
		pixNorm[row][col][blk] = Math<T>::mul(pixNorm[row][col][blk],
						      inv_frames);
	      }
	    }
	  }

#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int blk = 0; blk < BlockTileSize; ++blk) {
		pixNorm[row][col][blk] = Math<T>::reduceAdd(pixNorm[row][col][blk]);
	      }
	    }
	  }
	} // if-else NormLoop

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
		int smemRowIdx = row;
		int smemColIdx = col * RowTileSize;
		int smemBlockIdx = blk * ColTileSize * RowTileSize;
		int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		int smemIdx = smemBatchIdx * numWarps + warpId;
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
	      int smemRowIdx = row;
	      int smemColIdx = col * RowTileSize;
	      int smemBlockIdx = blk * ColTileSize * RowTileSize;
	      int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
	      int smemIdx = smemBatchIdx * numWarps + laneId;
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
		  if (outRow < vals.getSize(1) && outCol < vals.getSize(2) && outBlock < vals.getSize(0)) {
		    vals[outBlock][outRow][outCol] = NormSquared ?
		      ConvertTo<float>::to(pixNorm[row][col][blk])
		      : sqrtf(ConvertTo<float>::to(pixNorm[row][col][blk]));
		  }
		} else {
		  vals[outBlock][outRow][outCol] = NormSquared
		    ? ConvertTo<float>::to(pixNorm[row][col][blk])
		    : sqrtf(ConvertTo<float>::to(pixNorm[row][col][blk]));
		}
	      }
	    }
	  }
	}
      }
    
    }

    template <typename T>
    void runKmBurstL2Norm(Tensor<T, 5, true, int>& centroids,
			  Tensor<T, 4, true, int>& ave,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 3, true, int>& vals,
			  int patchsize, int nblocks,
			  bool normSquared,
			  cudaStream_t stream) {
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      constexpr int rowTileSize = 2;
      constexpr int colTileSize = 2;
      constexpr int blockTileSize = 1;

#define RUN_KMB_L2_ROW_MAJOR(TYPE_T, BURST)				\
      do {								\
        if (normLoop) {							\
	  if (normSquared) {						\
	    runKmBurstL2NormKernel<					\
										  TYPE_T, \
	      rowTileSize,						\
	      colTileSize,						\
	      blockTileSize,						\
										    true, \
										    true><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
	  } else {							\
	    runKmBurstL2NormKernel<					\
										  TYPE_T, \
										    rowTileSize, \
										    colTileSize, \
										    blockTileSize, \
										    true, \
										    false><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
	  }								\
        } else {							\
	  if (normSquared) {						\
	    runKmBurstL2NormKernel<					\
										TYPE_T,	\
										  rowTileSize, \
										  colTileSize, \
										  blockTileSize, \
										  false, \
										  true><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
	  } else {							\
	    runKmBurstL2NormKernel<					\
										  TYPE_T, \
										    rowTileSize, \
										    colTileSize, \
										    blockTileSize, \
										    false, \
										    false><<<grid, block, smem, stream>>>(BURST, ave, blocks, vals, patchsize, nblocks, TVecMax); \
	  }								\
        }								\
      } while (0)


      // compute numThreads
      int nftrs = centroids.getSize(0);
      int nframes = centroids.getSize(1);
      int dim = patchsize*patchsize*nftrs*nframes;
      bool normLoop = dim > maxThreads;
      int numThreads = std::min(dim, maxThreads);
      int nWarps = utils::divUp(numThreads, kWarpSize);
      // numThreads = utils::roundUp(numThreads,kWarpSize); // round-up for warp reduce.
      FAISS_ASSERT(vals.getSize(0) >= blocks.getSize(2));

      // compute number of Grids
      int blockBatchSize = vals.getSize(0);
      int height = vals.getSize(1);
      int width = vals.getSize(2);
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
      RUN_KMB_L2_ROW_MAJOR(T, centroids);

#undef RUN_NNF_L2

      CUDA_TEST_ERROR();
    }

    void runKmBurstL2Norm(
			  Tensor<float, 5, true>& centroids,
			  Tensor<float, 4, true>& ave,
			  Tensor<int, 5, true>& blocks,
			  Tensor<float, 3, true>& vals,
			  int patchsize,
			  int nblocks,
			  bool normSquared,
			  cudaStream_t stream) {
      runKmBurstL2Norm<float>(centroids, ave, blocks, vals,
			      patchsize, nblocks, normSquared, stream);
    }

    void runKmBurstL2Norm(
			  Tensor<half, 5, true>& centroids,
			  Tensor<half, 4, true>& ave,
			  Tensor<int, 5, true>& blocks,
			  Tensor<float, 3, true>& vals,
			  int patchsize,
			  int nblocks,
			  bool normSquared,
			  cudaStream_t stream) {
      runKmBurstL2Norm<half>(centroids, ave, blocks, vals,
			     patchsize, nblocks, normSquared, stream);
    }

  } // namespace gpu
} // namespace faiss
