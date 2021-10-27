
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/PairwiseDistances.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>

/***
    "indices" are the [y,x] or [row,col] 
    coorinates of the proposed iid patch
 ***/

namespace faiss {
  namespace gpu {

    __forceinline__ __device__ int hw_boundary(int hw, int max){
      hw = (hw) < 0 ? -(hw) : (hw);
      hw = (hw) > (max) ? (2*(max) - (hw)) : (hw);
      return hw;
    }

    __forceinline__ __device__ void get_top_left(Tensor<int, 5, true, int> indices,
				 int psHalf, int t0, int t1,
				 int blk, int row, int col,
				 int& row_t0, int& col_t0,
				 int& row_t1, int& col_t1){
      row_t0 = indices[0][t0][blk][row][col] - psHalf;
      col_t0 = indices[1][t0][blk][row][col] - psHalf;
      row_t1 = indices[0][t1][blk][row][col] - psHalf;
      col_t1 = indices[1][t1][blk][row][col] - psHalf;
    }
    __forceinline__ __device__ void get_indices(int tidx, int nftrs, int patchsize,
				int nframes, int& ftr, int& wIdx, int& hIdx){
      // features 
      ftr = tidx % nftrs;
      int fDiv = tidx / nftrs;

      // width
      wIdx = fDiv % patchsize;
      int wDiv = fDiv / patchsize;

      // height
      hIdx = wDiv % patchsize;
    }

    template <typename T, int hTile, int wTile, int bTile, int maxFrames, bool NormLoop>
    __global__ void pairwise_distances_kernel(Tensor<T, 5, true, int> dists,
					      Tensor<T, 4, true, int> burst,
					      Tensor<int, 5, true, int> indices,
					      Tensor<T, 5, true, int> centroids,
					      Tensor<int, 4, true, int> clusters,
					      int patchsize, float offset){


      //
      // compute pairwise distances across time.
      //

      // CUDA shared vars 
      extern __shared__ char smemByte[]; // nWarps*RowTileSize*ColTileSize*BlockTileSize 
      float* smem = (float*)smemByte;

      // CUDA vars
      int numWarps = utils::divUp(blockDim.x, kWarpSize);
      int laneId = getLaneId();
      int threadId = threadIdx.x;
      int warpId = threadId / kWarpSize;

      // get burst image sizes
      int nftrs = burst.getSize(0);
      int nframes = burst.getSize(1);
      int height = burst.getSize(2);
      int width = burst.getSize(3);
      int nindices = indices.getSize(2);
      int batchHeight = dists.getSize(3);
      int batchWidth = dists.getSize(4);

      // start of thread's batch
      int hStart = hTile * (blockIdx.x);
      int wStart = wTile * (blockIdx.y);
      int bStart = bTile * blockIdx.z;
      
      // vars for comp
      bool valid;
      int psHalf = patchsize/2;
      int dim = nftrs * patchsize * patchsize;
      T inv_dim = 1./dim;
      int h_t0,w_t0,h_t1,w_t1;
      int fIdx,row,col,blk,wOffset,hOffset;
      T burst_t0,burst_t1,delta;
      bool lastTile = false;
      int row_t0, col_t0, row_t1, col_t1;
      row_t0 = 0;
      col_t0 = 0;
      row_t1 = 0;
      col_t1 = 0;
      
      // accumulate in f32
      float thread_norm[hTile][wTile][bTile];

      //
      // compute average across patch for fixed (t0,t1) pair
      //
#pragma unroll
      for (int t0 = 0; t0 < maxFrames; ++t0) {
	if(t0 >= nframes){
	  continue;
	}
#pragma unroll
	for(int t1 = 0; t1 < maxFrames; ++t1){
	  if(t1 >= t0){
	    continue;
	  }


	  if (lastTile){
	    int numH = height - hStart;
	    int numW = width - wStart;
	    int numB = nindices - bStart;
	    for (int row_iter = 0; row_iter < numH; ++row_iter) {
	      for (int col_iter = 0; col_iter < numW; ++col_iter) {
		for (int blk_iter = 0; blk_iter < numB; ++blk_iter) {
		  // select (blk,row,col)
		  row = hStart + row_iter;
		  col = wStart + col_iter;
		  blk = bStart + blk_iter;

		  // sum across ps x ps x ftrs
		  if (NormLoop){
		    thread_norm[0][0][0] = 0;
		    for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x) {

		      // top-left indices
		      get_top_left(indices, psHalf, t0, t1, blk, row, col,
				   row_t0, col_t0, row_t1, col_t1);

		      // get (offsets & ftr) from thread index
		      get_indices(tIdx, nftrs, patchsize,
				  nframes, fIdx, wOffset, hOffset);

		      // valid hw indices
		      h_t0 = hw_boundary(row_t0 + hOffset, height-1);
		      w_t0 = hw_boundary(col_t0 + wOffset, width-1);
		      h_t1 = hw_boundary(row_t1 + hOffset, height-1);
		      w_t1 = hw_boundary(col_t1 + wOffset, width-1);

		      // compute delta 
		      burst_t0 = burst[fIdx][t0][h_t0][w_t0];
		      burst_t1 = burst[fIdx][t1][h_t1][w_t1];
		      delta = Math<T>::sub(burst_t0,burst_t1);
		      delta = Math<T>::mul(delta,delta);

		      // accumulate in local var
		      thread_norm[0][0][0] = thread_norm[0][0][0]
			+ Math<T>::reduceAdd(delta);
		    }
		  } else {

		    // set indices from thread index
		    get_indices(threadIdx.x, nftrs, patchsize,
				nframes, fIdx, wOffset, hOffset);

		    // top-left indices
		    get_top_left(indices, psHalf,t0, t1, blk, row, col,
				 row_t0, col_t0, row_t1, col_t1);


		    // valid hw indices
		    h_t0 = hw_boundary(row_t0 + hOffset, height-1);
		    w_t0 = hw_boundary(col_t0 + wOffset, width-1);
		    h_t1 = hw_boundary(row_t1 + hOffset, height-1);
		    w_t1 = hw_boundary(col_t1 + wOffset, width-1);

		    // compute delta 
		    burst_t0 = burst[fIdx][t0][h_t0][w_t0];
		    burst_t1 = burst[fIdx][t1][h_t1][w_t1];
		    delta = Math<T>::sub(burst_t0,burst_t1);
		    delta = Math<T>::mul(delta,delta);

		    // accumulate in local var
		    thread_norm[0][0][0] = Math<T>::reduceAdd(delta);
		  }

		  thread_norm[0][0][0] = warpReduceAllSum(thread_norm[0][0][0]);
		  if (laneId == 0) {
		    int smemRowIdx = row_iter;
		    int smemColIdx = col_iter * hTile;
		    int smemBlockIdx = blk_iter * wTile * hTile; 
		    int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		    int smemIdx = smemBatchIdx * numWarps + warpId;
		    smem[smemIdx] = thread_norm[0][0][0];
		  }

		} // blk_iter
	      } // col_iter
	    } // row_iter
	  }else{//lastTile

	    if (NormLoop){

	      T tmp[hTile][wTile][bTile];

	      // set thread vector to 0
#pragma unroll
	      for (int row_i = 0; row_i < hTile; ++row) {
#pragma unroll
		for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		  for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		    thread_norm[row_i][col_i][blk_i] = 0;
		  }
		}
	      }

	      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x) {
#pragma unroll
		for (int row_iter = 0; row_iter < hTile; ++row_iter) {
#pragma unroll
		  for (int col_iter = 0; col_iter < wTile; ++col_iter) {
#pragma unroll
		    for (int blk_iter = 0; blk_iter < bTile; ++blk_iter) {

		      // select (blk,row,col)
		      row = hStart + row_iter;
		      col = wStart + col_iter;
		      blk = bStart + blk_iter;

		      // set indices from thread index
		      get_indices(tIdx, nftrs, patchsize,
				  nframes, fIdx, wOffset, hOffset);

		      // top-left indices
		      get_top_left(indices, psHalf,t0, t1, blk, row, col,
				   row_t0, col_t0, row_t1, col_t1);

		      // valid hw indices
		      h_t0 = hw_boundary(row_t0 + hOffset, height-1);
		      w_t0 = hw_boundary(col_t0 + wOffset, width-1);
		      h_t1 = hw_boundary(row_t1 + hOffset, height-1);
		      w_t1 = hw_boundary(col_t1 + wOffset, width-1);

		      // compute delta 
		      burst_t0 = burst[fIdx][t0][h_t0][w_t0];
		      burst_t1 = burst[fIdx][t1][h_t1][w_t1];
		      tmp[row_iter][col_iter][blk_iter] = Math<T>::sub(burst_t0,burst_t1);
		    }
		  }
		}

		// compute squared
#pragma unroll
		for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		  for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		    for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		      thread_norm[row_i][col_i][blk_i] =
			Math<T>::mul(tmp[row_i][col_i][blk_i],
				     tmp[row_i][col_i][blk_i]);
		    }
		  }
		}

		// format
#pragma unroll
		for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		  for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		    for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		      thread_norm[row_i][col_i][blk_i] =
			Math<T>::reduceAdd(thread_norm[row_i][col_i][blk_i]);
		    }
		  }
		}
	      } // for ... threadIdx 
	    } else { // NormLoop=false

#pragma unroll
	      for (int row_iter = 0; row_iter < hTile; ++row_iter) {
#pragma unroll
		for (int col_iter = 0; col_iter < wTile; ++col_iter) {
#pragma unroll
		  for (int blk_iter = 0; blk_iter < bTile; ++blk_iter) {

		    // select (blk,row,col)
		    row = hStart + row_iter;
		    col = wStart + col_iter;
		    blk = bStart + blk_iter;

		    // top-left indices
		    get_top_left(indices, psHalf, t0, t1, blk, row, col,
		    		 row_t0, col_t0, row_t1, col_t1);
		    
		    // set indices from thread index
		    get_indices(threadIdx.x, nftrs, patchsize,
		    		nframes, fIdx, wOffset, hOffset);

		    // valid hw indices
		    h_t0 = hw_boundary(row_t0 + hOffset, height-1);
		    w_t0 = hw_boundary(col_t0 + wOffset, width-1);
		    h_t1 = hw_boundary(row_t1 + hOffset, height-1);
		    w_t1 = hw_boundary(col_t1 + wOffset, width-1);

		    // compute delta 
		    burst_t0 = burst[fIdx][t0][h_t0][w_t0];
		    burst_t1 = burst[fIdx][t1][h_t1][w_t1];
		    thread_norm[row_iter][col_iter][blk_iter] =
		      Math<T>::sub(burst_t0,burst_t1);
		  }
		}
	      }

	      // compute squared
#pragma unroll
	      for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		  for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		    thread_norm[row_i][col_i][blk_i] =
		      Math<T>::mul(thread_norm[row_i][col_i][blk_i],
				   thread_norm[row_i][col_i][blk_i]);
		  }
		}
	      }

	      // format
#pragma unroll
	      for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		  for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		    thread_norm[row_i][col_i][blk_i] =
		      Math<T>::reduceAdd(thread_norm[row_i][col_i][blk_i]);
		  }
		}
	      }

	    }

	    // Sum up all parts within each warp
#pragma unroll
	    for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
	      for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		  thread_norm[row_i][col_i][blk_i] =
		    warpReduceAllSum(thread_norm[row_i][col_i][blk_i]);
		}
	      }
	    }

	    if (laneId == 0) {
#pragma unroll
	      for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		  for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		    int smemRowIdx = row_i;
		    int smemColIdx = col_i * hTile;
		    int smemBlockIdx = blk_i * hTile * wTile;
		    int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		    int smemIdx = smemBatchIdx * numWarps + warpId;
		    smem[smemIdx] = thread_norm[row_i][col_i][blk_i];
		  }
		}
	      }
	    }
	
	  } // if(lastTile){}else{}
     
	  __syncthreads(); // sync threads across block

	  //
	  // sum across nWarps using first warp of threads
	  //
	  if (warpId == 0) {

	    // store each warp sum into thread of first warp
#pragma unroll
	    for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
	      for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		  int smemRowIdx = row_i;
		  int smemColIdx = col_i * hTile;
		  int smemBlockIdx = blk_i * wTile * hTile;
		  int smemBatchIdx = smemBlockIdx + smemRowIdx + smemColIdx;
		  int smemIdx = smemBatchIdx * numWarps + laneId;
		  thread_norm[row_i][col_i][blk_i] =
		    laneId < numWarps ? smem[smemIdx] : 0;
		}
	      }
	    }

	    // sum across first warp
#pragma unroll
	    for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
	      for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		  thread_norm[row_i][col_i][blk_i] =
		    warpReduceAllSum(thread_norm[row_i][col_i][blk_i]);
		}
	      }
	    }

	    // normalize
#pragma unroll
	    for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
	      for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		  thread_norm[row_i][col_i][blk_i] =
		    Math<T>::mul(thread_norm[row_i][col_i][blk_i],inv_dim);
		}
	      }
	    }


	    // Write out answer
	    if (laneId == 0) {
#pragma unroll
	      for (int row_i = 0; row_i < hTile; ++row_i) {
#pragma unroll
		for (int col_i = 0; col_i < wTile; ++col_i) {
#pragma unroll
		  for (int blk_i = 0; blk_i < bTile; ++blk_i) {
		    row = hStart + row_i;
		    col = wStart + col_i;
		    blk = bStart + blk_i;
		    if (lastTile) {
		      bool valid = row < height;
		      valid = valid && (col < width);
		      valid = valid && (blk < nindices);
		      if (valid){
			dists[t0][t1][blk][row][col] = thread_norm[row_i][col_i][blk_i];
		      }
		    } else {
		      dists[t0][t1][blk][row][col] = thread_norm[row_i][col_i][blk_i];
		    }

		  } // blk_i for-loop
		} // col_i for-loop
	      } // row_i for-loop
	    } // landId == 0
	  } // warpId == 0

	  __syncthreads(); // sync threads across block

	} // t1 for-loop
      } // t0 for-loop

    } 
    

    template <typename T>
    void pairwise_distances(Tensor<T, 5, true, int>& dists,
			    Tensor<T, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<T, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, float offset,
			    cudaStream_t stream){



      // create tiles
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      constexpr int hTile = 1;
      constexpr int wTile = 1;
      constexpr int bTile = 1;
      constexpr int maxFrames = 50;

      // compute numThreads
      int nftrs = burst.getSize(0);
      int nframes = burst.getSize(1);
      int dim = patchsize*patchsize*nftrs;
      bool normLoop = dim > maxThreads;
      int numThreads = std::min(dim, maxThreads);
      int nWarps = utils::divUp(numThreads, kWarpSize);

      // compute number of Grids
      int height = dists.getSize(3);
      int width = dists.getSize(4);
      int nblocks = blocks.getSize(2);
      int numToComp = height * width * nblocks * nframes;
      int numToCompPerKernel = hTile * wTile * nblocks;
      int numHeightBlocks = utils::divUp(height, hTile);
      int numWidthBlocks = utils::divUp(width, wTile);
      int numPixBlocks = numHeightBlocks * numWidthBlocks;
      int numBlockBlocks = utils::divUp(nblocks, bTile);
      int nBlocks = utils::divUp(numToComp,numToCompPerKernel);

      // launch config
      auto grid = dim3(numHeightBlocks,numWidthBlocks,numBlockBlocks);
      auto block = dim3(numThreads);
      auto smem = sizeof(float) * numToCompPerKernel * nWarps;

      // launch
      if (normLoop){
	pairwise_distances_kernel<T,hTile,wTile,bTile,maxFrames,true>
	  <<<grid,block,smem,stream>>>
	  (dists,burst,blocks,centroids,clusters,patchsize,offset);
      }else{
	pairwise_distances_kernel<T,hTile,wTile,bTile,maxFrames,false>
	  <<<grid,block,smem,stream>>>
	  (dists,burst,blocks,centroids,clusters,patchsize,offset);
      }
							    
    }




    //
    // Template Inits
    //
    
    void pairwise_distances(Tensor<float, 5, true, int>& dists,
			    Tensor<float, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<float, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, float offset,
			    cudaStream_t stream){
      pairwise_distances<float>(dists,burst,blocks,centroids,clusters,
				patchsize, offset, stream);
    }

    void pairwise_distances(Tensor<half, 5, true, int>& dists,
			    Tensor<half, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<half, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, float offset,
			    cudaStream_t stream){
      pairwise_distances<half>(dists,burst,blocks,centroids,clusters,
			       patchsize, offset, stream);
    }




  }
}