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
	Tensor<T, 3, true> subAve,
	Tensor<int, 5, true> blocks,
	Tensor<bool, 4, true> mask,
        Tensor<T, 4, true> ave,
	int total_nframes,
	int patchsize,
	int nblocks) {

    // shapes 
    int sub_nframes = burst.getSize(0); 
    int nftrs = ave.getSize(0);
    int blockBatchSize = ave.getSize(1);
    int height = ave.getSize(2);
    int width = ave.getSize(3);
    int heightPad = burst.getSize(2);
    int widthPad = burst.getSize(3);
    int nbHalf = nblocks/2;
    int psHalf = patchsize/2;
    T zero = ConvertTo<T>::to(0);

    // block indices
    int ftrStart = FtrTileSize*(blockIdx.x);
    int rowStart = RowTileSize*(blockIdx.y);
    int colStart = ColTileSize*(blockIdx.z);
    int rowStartBurst = rowStart + nbHalf;// + psHalf;
    int colStartBurst = colStart + nbHalf;// + psHalf;

    // determine if our batchsize is too big for the location;
    bool lastFtrTile = (ftrStart + FtrTileSize - 1) >= ave.getSize(0);
    bool lastRowTile = (rowStart + RowTileSize - 1) >= ave.getSize(2);
    bool lastColTile = (colStart + ColTileSize - 1) >= ave.getSize(3);
    bool lastTile = lastFtrTile || lastRowTile || lastColTile;
    T inv_nframes = 1. / (total_nframes*1.0);
    T sum[FtrTileSize][RowTileSize][ColTileSize];
    // printf("(rowStart, colStart): (%d,%d)\n",rowStart,colStart);
    // if ((rowStart == 16 ) && (colStart == 16)){
    // 	printf("pix 16,16!\n");
    // }
    // if ((rowStart == 25 ) && (colStart == 25)){
    // 	printf("pix 25,25!\n");
    // }
    // if ((rowStart == 28 ) && (colStart == 28)){
    // 	printf("pix 28,28!\n");
    // }
    // if ((rowStart == 31 ) && (colStart == 31)){
    // 	printf("pix 31,31!\n");
    // }
    
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

		for (int blk = threadIdx.x; blk < blockBatchSize; blk += blockDim.x) {

		  int ftrIndex = ftrStart+ftr;
		  int a_row = rowStart+row;
		  int a_col = colStart+col;

		  for (int tidx = 0; tidx < sub_nframes; ++tidx){

		    int blRow = blocks[blk][a_row][a_col][tidx][0];
		    int blCol = blocks[blk][a_row][a_col][tidx][1];
		    bool m_val = true;//mask[blk][a_row][a_col][tidx];

		    int b_row = rowStartBurst + blRow + row;
		    int b_col = colStartBurst + blCol + col;

		    T b_val = m_val ? burst[tidx][ftrIndex][b_row][b_col] : zero;
		    T s_val = sum[0][0][0];
		    sum[0][0][0] = Math<T>::add(b_val,s_val);
		  }
		  sum[0][0][0] = Math<T>::mul(sum[0][0][0],inv_nframes);
		  ave[ftrIndex][blk][a_row][a_col]
		    = Math<T>::add(sum[0][0][0],subAve[ftrIndex][a_row][a_col]);
		}

	      }else {
		  
		int ftrIndex = ftrStart+ftr;
		int a_row = rowStart+row;
		int a_col = colStart+col;

		for (int tidx = 0; tidx < sub_nframes; ++tidx){
		  
		  int blRow = blocks[threadIdx.x][a_row][a_col][tidx][0];
		  int blCol = blocks[threadIdx.x][a_row][a_col][tidx][1];
		  bool m_val = true;//mask[threadIdx.x][a_row][a_col][tidx];

		  int b_row = rowStartBurst+ blRow +row;
		  int b_col = colStartBurst+ blCol +col;

		  T b_val = m_val ? burst[tidx][ftrIndex][b_row][b_col] : zero;
		  T s_val = sum[0][0][0];
		  sum[0][0][0] = Math<T>::add(b_val,s_val);
		}
		sum[0][0][0] = Math<T>::mul(sum[0][0][0],inv_nframes);
		ave[ftrIndex][threadIdx.x][a_row][a_col]
		  = Math<T>::add(sum[0][0][0],subAve[ftrIndex][a_row][a_col]);
		// ave[ftrIndex][threadIdx.x][a_row][a_col] = sum[0][0][0];

	      }
	      }
	    }
	  }
    } else {
      
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

      if (NormLoop) {

	for (int blk = threadIdx.x; blk < blockBatchSize; blk += blockDim.x) {
	  for (int tidx = 0; tidx < sub_nframes; ++tidx){
#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int ftr = 0; ftr < FtrTileSize; ++ftr) {

		  int a_row = rowStart + row;
		  int a_col = colStart + col;

		  int blRow = blocks[blk][a_row][a_col][tidx][0];
		  int blCol = blocks[blk][a_row][a_col][tidx][1];
		  bool m_val = true;//mask[blk][a_row][a_col][tidx];

		  int b_ftr = ftrStart+ftr;
		  int b_row = rowStartBurst + blRow + row;
		  int b_col = colStartBurst + blCol + col;

		  T b_val = m_val ? burst[tidx][b_ftr][b_row][b_col] : zero;
		  T s_val = sum[ftr][row][col];
		  sum[ftr][row][col] = Math<T>::add(b_val,s_val);
		}
	      }
	    }

#pragma unroll
	    for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	      for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
		for (int ftr = 0; ftr < FtrTileSize; ++ftr) {
		  int a_ftr = ftrStart + ftr;
		  int a_row = rowStart + row;
		  int a_col = colStart + col;

		  T temp = Math<T>::mul(sum[ftr][row][col],inv_nframes);
		  temp = Math<T>::add(temp,subAve[a_ftr][a_row][a_col]);
		  ave[a_ftr][blk][a_row][a_col] = temp;
		}
	      }
	    }
	  }
	}
	  
      } else {

	// A block of threads is the exact size of the vector
	for (int tidx = 0; tidx < sub_nframes; ++tidx){
#pragma unroll
	  for (int row = 0; row < RowTileSize; ++row) {
#pragma unroll
	    for (int col = 0; col < ColTileSize; ++col) {
#pragma unroll
	      for (int ftr = 0; ftr < FtrTileSize; ++ftr) {

		int a_row = rowStart + row;
		int a_col = colStart + col;
		int blRow = blocks[threadIdx.x][a_row][a_col][tidx][0];
		int blCol = blocks[threadIdx.x][a_row][a_col][tidx][1];
		bool m_val = true;//mask[threadIdx.x][a_row][a_col][tidx];

		int b_ftr = ftrStart+ftr;
		int b_row = rowStartBurst+blRow+row;
		int b_col = colStartBurst+blCol+col;

		T b_val = m_val ? burst[tidx][b_ftr][b_row][b_col] : zero;
		T s_val = sum[ftr][row][col];
		sum[ftr][row][col] = Math<T>::add(b_val,s_val);
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
	      int a_ftr = ftrStart + ftr;
	      int a_row = rowStart + row;
	      int a_col = colStart + col;
	      T temp = Math<T>::mul(sum[ftr][row][col],inv_nframes);
	      temp = Math<T>::add(temp,subAve[a_ftr][a_row][a_col]);
	      ave[a_ftr][threadIdx.x][a_row][a_col] = temp;
	    }
	  }
	}
      }
    }
}

template <typename T>
void runSubBurstAverage(
	Tensor<T, 4, true>& burst,
	Tensor<T, 3, true>& subAve,
        Tensor<int, 5, true>& blocks,
	Tensor<bool, 4, true>& mask,
	Tensor<T, 4, true>& ave,
	int total_nframes,
	int patchsize,
	int nblocks,
        cudaStream_t stream) {

    // size checking
    FAISS_ASSERT(burst.getSize(0) == blocks.getSize(3)); // sub_nframes
    FAISS_ASSERT(burst.getSize(1) == ave.getSize(0)); // nftrs
    // FAISS_ASSERT(burst.getSize(2) == ave.getSize(2)); // height
    // FAISS_ASSERT(burst.getSize(3) == ave.getSize(3)); // width
    FAISS_ASSERT(blocks.getSize(1) == ave.getSize(2)); // height
    FAISS_ASSERT(blocks.getSize(2) == ave.getSize(3)); // width
    FAISS_ASSERT(ave.getSize(1) == blocks.getSize(0)); // nblocks
    FAISS_ASSERT(blocks.getSize(4) == 2); // (x,y)

    // constants
    constexpr int FtrTileSize = 1;
    constexpr int RowTileSize = 2;
    constexpr int ColTileSize = 2;

    // shapes 
    int blockBatchSize = blocks.getSize(0);
    int sub_nframes = burst.getSize(0);
    int heightB = burst.getSize(2);
    int widthB = burst.getSize(3);
    int nftrs = ave.getSize(0);
    int height = ave.getSize(2);
    int width = ave.getSize(3);

    // printf("(nftrs, height, width): (%d, %d, %d)\n",nftrs,height,width);
    // printf("(nframes, nblocks, two): (%d, %d, %d)\n",nframes,nblocks,blocks.getSize(2));
    // printf("(heightB, widthB): (%d, %d)\n",heightB,widthB);
    // threads and blocks
    int threadsPerBlock = std::min(blockBatchSize, getMaxThreadsCurrentDevice());
    bool NormLoop = blockBatchSize > threadsPerBlock;
    int FtrGrid = utils::divUp(nftrs, FtrTileSize);
    int RowGrid = utils::divUp(height, RowTileSize);
    int ColGrid = utils::divUp(width, ColTileSize);

    // launch cuda kernel
    auto block = dim3(threadsPerBlock);
    auto grid = dim3(FtrGrid,RowGrid,ColGrid);
    if (NormLoop) {
      averageForEachBlock<T, FtrTileSize, RowTileSize, ColTileSize, true>
	<<<grid, block, 0, stream>>>(burst, subAve, blocks, mask, ave,
				     total_nframes, patchsize, nblocks);
    } else {
      averageForEachBlock<T, FtrTileSize, RowTileSize, ColTileSize, false>
	<<<grid, block, 0, stream>>>(burst, subAve, blocks, mask, ave,
				     total_nframes, patchsize, nblocks);
    }

    CUDA_TEST_ERROR();
}

void runSubBurstAverage(
	Tensor<float, 4, true>& burst,
	Tensor<float, 3, true>& subAve,
        Tensor<int, 5, true>& blocks,
	Tensor<bool, 4, true> mask,
	Tensor<float, 4, true>& ave,
	int total_nframes,
	int patchsize,
	int nblocks,
        cudaStream_t stream) {
  runSubBurstAverage<float>(burst, subAve, blocks, mask, ave,
			    total_nframes, patchsize, nblocks, stream);
}

void runSubBurstAverage(
	Tensor<half, 4, true>& burst,
	Tensor<half, 3, true>& subAve,
        Tensor<int, 5, true>& blocks,
	Tensor<bool, 4, true> mask,
	Tensor<half, 4, true>& ave,
	int total_nframes,
	int patchsize,
	int nblocks,
        cudaStream_t stream) {
  runSubBurstAverage<half>(burst, subAve, blocks, mask, ave,
			   total_nframes, patchsize, nblocks, stream);
}

} // namespace gpu
} // namespace faiss
