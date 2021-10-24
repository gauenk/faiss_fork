

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmClusterUpdate.cuh>
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
    __global__ void init_centroids_kernel(Tensor<T, 5, true, int> dists,
					    Tensor<T, 4, true, int> burst,
					    Tensor<int, 5, true, int> blocks,
					    Tensor<T, 5, true, int> centroids,
					    Tensor<int, 4, true, int> clusters,
					    Tensor<int, 1, true, int> sizes){
      // compute pairwise distances across time.
      
    }



    template <typename T>
    __global__ void update_centroids_kernel(Tensor<T, 5, true, int> dists,
					    Tensor<T, 4, true, int> burst,
					    Tensor<int, 5, true, int> blocks,
					    Tensor<T, 5, true, int> centroids,
					    Tensor<int, 4, true, int> clusters,
					    Tensor<int, 1, true, int> sizes){
      // compute pairwise distances across time.
      
    }



    template <typename T>
    void update_centroids(Tensor<T, 5, true, int>& dists,
			  Tensor<T, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<T, 5, true, int>& centroids,
			  Tensor<int, 4, true, int>& clusters,
			  Tensor<int, 1, true, int>& sizes,
			  bool init_update, cudaStream_t stream){


      // threads 
      int maxThreads = 1;
      int dim = 1;
      int numThreads = std::min(dim, maxThreads);

      // blocks
      int numGrids = 1;

      // launch
      if (init_update){
	auto grid = dim3(numGrids);
	auto block = dim3(numThreads);
	init_centroids_kernel<T><<<grid,block,0,stream>>>(dists,burst,blocks,
							  centroids,clusters,sizes);
      }else{
	auto grid = dim3(numGrids);
	auto block = dim3(numThreads);
	update_centroids_kernel<T><<<grid,block,0,stream>>>(dists,burst,blocks,
							    centroids,clusters,sizes);
      }
							    
    }


    //
    // Template Inits
    //
    void update_centroids(Tensor<float, 5, true, int>& dists,
			  Tensor<float, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 5, true, int>& centroids,
			  Tensor<int, 4, true, int>& clusters,
			  Tensor<int, 1, true, int>& sizes,
			  bool init_update, cudaStream_t stream){
      update_centroids<float>(dists,burst,blocks,centroids,
			      clusters,sizes,init_update,stream);
    }

    void update_centroids(Tensor<half, 5, true, int>& dists,
			  Tensor<half, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<half, 5, true, int>& centroids,
			  Tensor<int, 4, true, int>& clusters,
			  Tensor<int, 1, true, int>& sizes,
			  bool init_update, cudaStream_t stream){
      update_centroids<half>(dists,burst,blocks,centroids,
			     clusters,sizes,init_update,stream);
    }


  }
}
