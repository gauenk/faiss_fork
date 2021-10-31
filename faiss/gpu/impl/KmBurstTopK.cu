
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/ComputeModes.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <algorithm>


namespace faiss {
  namespace gpu {

#define ABS(N) (((N)<0)?(-(N)):((N)))

    /*********************************************

           Pick topK from nblocks

    *********************************************/

    template <typename T> __global__ 
    void kmb_topK_kernel(Tensor<float, 3, true, int> dists,
			 Tensor<int, 5, true, int> inds,
			 Tensor<int, 5, true, int> outInds,
			 Tensor<float, 3, true, int> outDists,
			 Tensor<T, 3, true, int> modes){
      // unpack shapes 
      int nframes = inds.getSize(1);
      int nblocks = inds.getSize(2);
      int height = inds.getSize(3);
      int width = inds.getSize(4);
      int K = outInds.getSize(2);

      // index using blocks: number of blocks is _always_ exactly equal to threads
      int h = threadIdx.x + blockDim.x * blockIdx.x;
      int w = threadIdx.y + blockDim.y * blockIdx.y;
      bool legal_h = h < height;
      bool legal_w = w < width;
      bool legal = legal_h && legal_w;

      // helpers
      float val,val_max,val_curr,fmode;
      int kidx;
      
      // continue of legal
      if (legal){

	// set init
	val_max = (float)outDists[K-1][h][w];
	val_curr = val_max;
	for (int blk = 0; blk < nblocks; ++blk){
	  fmode = ConvertTo<float>::to(modes[blk][h][w]);
	  val = (float)ABS(Math<float>::sub(dists[blk][h][w],fmode));

	  if (val < val_max){
	    kidx = K-1;
	    val_curr = val_max;
	    while( val < val_curr && kidx > 0){
	      kidx -= 1;
	      val_curr = outDists[kidx][h][w];
	    }
	    if (kidx != 0){ kidx += 1; }
	    else if (val > val_curr){ kidx += 1; }

	    // shift values up
	    for (int sidx = K-1; sidx > kidx; --sidx){
	      outDists[sidx][h][w] = (float)outDists[sidx-1][h][w];
	      for (int fidx = 0; fidx < nframes; ++fidx){
	  	outInds[0][fidx][sidx][h][w] =
	  	  (int) outInds[0][fidx][sidx-1][h][w];
	  	outInds[1][fidx][sidx][h][w] =
	  	  (int) outInds[1][fidx][sidx-1][h][w];
	      }
	    }

	    // assign new values
	    outDists[kidx][h][w] = val;
	    for (int fidx = 0; fidx < nframes; ++fidx){
	      outInds[0][fidx][kidx][h][w] = (int)inds[0][fidx][blk][h][w];
	      outInds[1][fidx][kidx][h][w] = (int)inds[1][fidx][blk][h][w];
	    }
	    val_max = outDists[K-1][h][w];

	  }
	}
      }

    }

    template <typename T>
    void kmb_topK(Tensor<float, 3, true, int> dists,
		  Tensor<int, 5, true, int> blocks,
		  Tensor<int, 5, true, int> outInds,
		  Tensor<float, 3, true, int> outDists,
		  Tensor<T, 3, true, int> modes,
		  cudaStream_t stream){

      // shapes
      int two = blocks.getSize(0);
      int nframes = blocks.getSize(1);
      int bBatch = blocks.getSize(2);
      int hBatch = blocks.getSize(3);
      int wBatch = blocks.getSize(4);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int numThreads = 1;
      int numThreadsTotal = numThreads*numThreads;
      FAISS_ASSERT(numThreadsTotal <= maxThreads);

      // create grid 
      int hBlocks = utils::divUp(hBatch,numThreads);
      int wBlocks = utils::divUp(wBatch,numThreads);

      // launch params
      auto grid = dim3(hBlocks,wBlocks);
      auto block = dim3(numThreads,numThreads);

      // launch kernel
      kmb_topK_kernel<<<grid,block,0,stream>>>
	(dists,blocks,outInds,outDists,modes);

      // error check
      CUDA_TEST_ERROR();
    }


    void kmb_topK(Tensor<float, 3, true, int> dists,
		  Tensor<int, 5, true, int> blocks,
		  Tensor<int, 5, true, int> outInds,
		  Tensor<float, 3, true, int> outDists,
		  Tensor<float, 3, true, int> modes,
		  cudaStream_t stream){
      kmb_topK<float>(dists,blocks,outInds,outDists,modes,stream);
    }

    void kmb_topK(Tensor<float, 3, true, int> dists,
		  Tensor<int, 5, true, int> blocks,
		  Tensor<int, 5, true, int> outInds,
		  Tensor<float, 3, true, int> outDists,
		  Tensor<half, 3, true, int> modes,
		  cudaStream_t stream){
      kmb_topK<half>(dists,blocks,outInds,outDists,modes,stream);
    }


  } // namespace gpu
} // namespace faiss
