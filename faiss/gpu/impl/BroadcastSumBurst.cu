/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <algorithm>

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

#define inline_min(a, b) ( (a) < (b) ? (a) : (b) )

template <typename T,
    int FtrTileSize,
    int RowTileSize,
    int ColTileSize,
    bool NormLoop>
__global__ void averageForEachBlock(
	Tensor<T, 4, true> burst,
	Tensor<int, 3, true> blocks,
        Tensor<T, 4, true> ave) {

    // shapes 
    int nframes = burst.getSize(0); 
    int nftrs = ave.getSize(0);
    int nblocks = ave.getSize(1);
    int height = ave.getSize(2);
    int width = ave.getSize(3);

    // block indices
    int ftrStart = blockIdx.x;
    int rowStart = blockIdx.y;
    int colStart = blockIdx.z;

    // determine if our batchsize is too big for the location;
    bool lastFtrTile = (ftrStart + FtrTileSize - 1) >= ave.getSize(0);
    bool lastRowTile = (rowStart + RowTileSize - 1) >= ave.getSize(2);
    bool lastColTile = (colStart + ColTileSize - 1) >= ave.getSize(3);
    bool lastTile = lastFtrTile || lastRowTile || lastColTile;
    float inv_nframes = 1. / (nframes*1.0);
    float sum[FtrTileSize][RowTileSize][ColTileSize];
    
    if (lastTile) { // our batchsizes are too big for the start location

        // We are handling the very end of the input matrix rows
        int numFtrIters = nftrs - ftrStart;
	int numRowIters = height - rowStart;
	int numColIters = width - colStart;
	numFtrIters = inline_min(FtrTileSize,(int)numFtrIters);
	numRowIters = inline_min(RowTileSize,(int)numRowIters);
	numColIters = inline_min(ColTileSize,(int)numColIters);

	for (int row = 0; row < numRowIters; ++row) {
	  for (int col = 0; col < numColIters; ++col) {
	    for (int ftr = 0; ftr < numFtrIters; ++ftr) {

	      if (NormLoop) {

		for (int blkIndex = threadIdx.x;
		     blkIndex < nblocks;
		     blkIndex += blockDim.x) {
		  for (int tidx = 0; tidx < nframes; ++tidx){
		    int b_row = blocks[tidx][blkIndex][0]+row;
		    int b_col = blocks[tidx][blkIndex][0]+col;

		    T b_val = burst[tidx][ftr][b_row][b_col];
		    T s_val = sum[0][0][0];
		    sum[0][0][0] = Math<float>::add(b_val,s_val);
		  }
		  ave[ftr][blkIndex][row][col] = Math<float>::mul(sum[0][0][0],
								  inv_nframes);
		}
	      }else {
		  
		  for (int tidx = 0; tidx < nframes; ++tidx){

		    int b_row = blocks[tidx][threadIdx.x][0]+row;
		    int b_col = blocks[tidx][threadIdx.x][0]+col;

		    T b_val = burst[tidx][ftr][b_row][b_col];
		    T s_val = sum[0][0][0];
		    sum[0][0][0] = Math<float>::add(b_val,s_val);
		  }
		  ave[ftr][threadIdx.x][row][col] = Math<float>::mul(sum[0][0][0],
								     inv_nframes);
		}
	      }
	    }
	  }
    } else {
      
      if (NormLoop) {

#pragma unroll
	for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	  for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	    for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
	      sum[ftr][row][col]  = 0;
	    }
	  }
	}

	for (int blk = threadIdx.x; blk < nblocks; blk += blockDim.x) {
	  for (int tidx = 0; tidx < nframes; ++tidx){
#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
		  int b_row = blocks[tidx][blk][0]+row;
		  int b_col = blocks[tidx][blk][0]+col;

		  T b_val = burst[tidx][ftr][b_row][b_col];
		  T s_val = sum[ftr][row][col];
		  sum[ftr][row][col] = Math<float>::add(b_val,s_val);
		}
	      }
	    }

#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
		  ave[ftr][blk][row][col] = Math<float>::mul(sum[ftr][row][col],
							     inv_nframes);
		}
	      }
	    }
	  }
	}
	  
      } else {

	// A block of threads is the exact size of the vector
	for (int tidx = 0; tidx < nframes; ++tidx){
#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
		int b_row = blocks[tidx][threadIdx.x][0]+row;
		int b_col = blocks[tidx][threadIdx.x][0]+col;

		T b_val = burst[tidx][ftr][b_row][b_col];
		T s_val = sum[ftr][row][col];
		sum[ftr][row][col] = Math<float>::add(b_val,s_val);
	      }
	    }
	  }
	}
#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
		ave[ftr][threadIdx.x][row][col] = Math<float>::mul(sum[ftr][row][col],
								     inv_nframes);
	      }
	    }
	  }
      }
    }
}

template <typename T>
void runBurstAverage(
	Tensor<T, 4, true>& burst,
        Tensor<int, 3, true>& blocks,
	Tensor<T, 4, true>& ave,
        cudaStream_t stream) {

    // size checking
    FAISS_ASSERT(burst.getSize(0) == blocks.getSize(0)); // nframes
    FAISS_ASSERT(burst.getSize(1) == ave.getSize(0)); // nftrs
    FAISS_ASSERT(burst.getSize(2) == ave.getSize(2)); // height
    FAISS_ASSERT(burst.getSize(3) == ave.getSize(3)); // width
    FAISS_ASSERT(ave.getSize(1) == blocks.getSize(1)); // nblocks
    FAISS_ASSERT(blocks.getSize(2) == 2); // (x,y)

    // constants
    constexpr int FtrTileSize = 2;
    constexpr int RowTileSize = 2;
    constexpr int ColTileSize = 2;

    // shapes 
    int nblocks = blocks.getSize(1);
    int nframes = burst.getSize(0);
    int nftrs = burst.getSize(1);
    int height = burst.getSize(2);
    int width = burst.getSize(3);

    // threads and blocks
    int threadsPerBlock = std::min(nblocks, getMaxThreadsCurrentDevice());
    bool NormLoop = nblocks > threadsPerBlock;
    int FtrGrid = utils::divUp(nftrs, FtrTileSize);
    int RowGrid = utils::divUp(height, RowTileSize);
    int ColGrid = utils::divUp(width, ColTileSize);

    // launch cuda kernel
    auto block = dim3(threadsPerBlock);
    auto grid = dim3(FtrGrid,RowGrid,ColGrid);
    if (NormLoop) {
      averageForEachBlock<T, FtrTileSize, RowTileSize, ColTileSize, true>
	<<<grid, block, 0, stream>>>(burst, blocks, ave);
    } else {
      averageForEachBlock<T, FtrTileSize, RowTileSize, ColTileSize, false>
	<<<grid, block, 0, stream>>>(burst, blocks, ave);
    }

    CUDA_TEST_ERROR();
}

void runBurstAverage(
	Tensor<float, 4, true>& burst,
        Tensor<int, 3, true>& blocks,
	Tensor<float, 4, true>& ave,
        cudaStream_t stream) {
   runBurstAverage<float>(burst, blocks, ave, stream);
}

void runBurstAverage(
	Tensor<half, 4, true>& burst,
        Tensor<int, 3, true>& blocks,
	Tensor<half, 4, true>& ave,
        cudaStream_t stream) {
   runBurstAverage<half>(burst, blocks, ave, stream);
}

} // namespace gpu
} // namespace faiss
