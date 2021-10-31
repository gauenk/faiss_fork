
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


    /*********************************************

           Compute Mode Centroids

    *********************************************/

    template <typename T>
    __global__ void compute_mode_centroids_kernel(float std, int patchsize, int nftrs,
						  Tensor<uint8_t, 4, true, int> sizes,
						  Tensor<T, 4, true, int> modes){

      // height,width
      int h = blockIdx.x;
      int w = blockIdx.y;

      // get dims to comp indices from thread
      int nclusters = sizes.getSize(0);
      int nblocks = sizes.getSize(1);
      int dim = nclusters*nblocks;

      // helpers
      bool gtz;
      int b,c0,num,size;
      T mode,val,svar,var_c0,n_ratio,numT,sizeT,var_c1;
      T var = std * std;

      // p ratio
      T psT = (T)patchsize;
      T nftrsT = (T)nftrs;
      T P = psT * psT * nftrsT;
      T p_ratio = Math<T>::sub(P,2)/P;

      // set clusters
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){

	c0 = tIdx % nclusters;
	b = (tIdx / nclusters) % nblocks;

	// sum vars
	num = 0;
	svar = 0;
	for (int c1 = 0; c1 < nclusters; ++c1){
	  size = sizes[c1][b][h][w];
	  gtz = size > 0;
	  gtz = gtz && (c1 != c0);
	  sizeT = (T)size;
	  var_c1 = var / sizeT;
	  val = gtz ? var_c1 : (T)0;
	  svar += val;
	  num = num + (gtz ? 1 : 0);
	}

	// ratio of num
	numT = (T)num;
	n_ratio = ( Math<T>::sub(numT,1) / numT );
	n_ratio = n_ratio * n_ratio;
	
	// variance of current centroid
	size = sizes[c0][b][h][w];
	sizeT = (T)size;
	var_c0 = var / sizeT;

	// compute mode 
	mode = n_ratio * var_c0 + svar / numT;
	mode = p_ratio * mode;

	if (size == 0){
	  modes[c0][b][h][w] = 0;
	}else{
	  modes[c0][b][h][w] = mode;
	}

      }
    }

    template <typename T>
    void compute_mode_centroids(float std, int patchsize, int nftrs,
				 Tensor<uint8_t, 4, true, int> sizes,
				 Tensor<T, 4, true, int> modes,
				 cudaStream_t stream) {

      // shapes
      int nclusters = sizes.getSize(0);
      int bBatch = sizes.getSize(1);
      int hBatch = sizes.getSize(2);
      int wBatch = sizes.getSize(3);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nclusters*bBatch;
      int numThreads = std::min(dim, maxThreads);

      // launch
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch kernel
      compute_mode_centroids_kernel
      	<<<grid,block,0,stream>>>(std,patchsize,nftrs,sizes,modes);

      // error check
      CUDA_TEST_ERROR();
    }

    void compute_mode_centroids(float std, int patchsize, int nftrs,
				Tensor<uint8_t, 4, true, int> sizes,
				Tensor<float, 4, true, int> modes,
				cudaStream_t stream){
      compute_mode_centroids<float>(std,patchsize,nftrs,sizes,modes,stream);
    }

    void compute_mode_centroids(float std, int patchsize, int nftrs,
				Tensor<uint8_t, 4, true, int> sizes,
				Tensor<half, 4, true, int> modes,
				cudaStream_t stream){
      compute_mode_centroids<half>(std,patchsize,nftrs,sizes,modes,stream);
    }

  } // namespace gpu
} // namespace faiss
