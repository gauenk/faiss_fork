
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


namespace faiss {
  namespace gpu {

    template <typename T>
    __global__ void pairwise_distances_kernel(Tensor<T, 5, true, int> dists,
					      Tensor<T, 4, true, int> burst,
					      Tensor<int, 5, true, int> blocks,
					      Tensor<T, 5, true, int> centroids,
					      Tensor<int, 4, true, int> clusters,
					      int patchsize){
      // compute pairwise distances across time.
      
    }



    template <typename T>
    void pairwise_distances(Tensor<T, 5, true, int>& dists,
			    Tensor<T, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<T, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, cudaStream_t stream){


      // threads 
      int maxThreads = 1;
      int dim = 1;
      int numThreads = std::min(dim, maxThreads);

      // blocks
      int numGrids = 1;

      // launch
      auto grid = dim3(numGrids);
      auto block = dim3(numThreads);
      pairwise_distances_kernel<T><<<grid,block,0,stream>>>(dists,burst,
							    blocks,centroids,
							    clusters,patchsize);
							    
    }




    //
    // Template Inits
    //
    
    void pairwise_distances(Tensor<float, 5, true, int>& dists,
			    Tensor<float, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<float, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, cudaStream_t stream){
      pairwise_distances<float>(dists,burst,blocks,centroids,clusters,
			 patchsize, stream);
    }

    void pairwise_distances(Tensor<half, 5, true, int>& dists,
			    Tensor<half, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<half, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, cudaStream_t stream){
      pairwise_distances<half>(dists,burst,blocks,centroids,clusters,
			 patchsize, stream);
    }




  }
}