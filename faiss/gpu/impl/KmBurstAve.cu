
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
    __global__ void kmb_ave_kernel(Tensor<T, 5, true, int> centroids,
				   Tensor<T, 4, true, int> ave){

      // height,width
      int h = blockIdx.x;
      int w = blockIdx.y;
      int b = blockIdx.z;

      // get dims to comp indices from thread
      int nftrs = centroids.getSize(0);
      int nclusters = centroids.getSize(1);
      int dim = nftrs;
      T inv_nclusters = 1./nclusters;

      // helpers
      int fIdx,cIdx;
      T ave_val;

      // set clusters
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){
	fIdx = tIdx % nftrs;

	ave_val = 0;
	for (int cIdx = 0; cIdx < nclusters; ++cIdx){
	  ave_val += centroids[fIdx][cIdx][b][h][w];
	}
	ave[fIdx][b][h][w] = Math<T>::mul(ave_val,inv_nclusters);

      }
    }

    template <typename T>
    void kmb_ave(Tensor<T, 5, true, int> centroids,
		 Tensor<T, 4, true, int> ave,
		 cudaStream_t stream){

      // shapes
      int nftrs = centroids.getSize(0);
      int nclusters = centroids.getSize(1);
      int bBatch = centroids.getSize(2);
      int hBatch = centroids.getSize(3);
      int wBatch = centroids.getSize(4);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nftrs;
      int numThreads = std::min(dim, maxThreads);

      // launch
      auto grid = dim3(hBatch,wBatch,bBatch);
      auto block = dim3(numThreads);

      // launch kernel
      kmb_ave_kernel<<<grid,block,0,stream>>>(centroids,ave);

      // error check
      CUDA_TEST_ERROR();
    }

    void kmb_ave(Tensor<float, 5, true, int> centroids,
		 Tensor<float, 4, true, int> ave,
		 cudaStream_t stream){
      kmb_ave<float>(centroids,ave,stream);
    }

    void kmb_ave(Tensor<half, 5, true, int> centroids,
		 Tensor<half, 4, true, int> ave,
		 cudaStream_t stream){
      kmb_ave<half>(centroids,ave,stream);
    }

  } // namespace gpu
} // namespace faiss
