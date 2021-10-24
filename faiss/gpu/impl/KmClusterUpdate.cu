

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
    __global__ void init_clusters_kernel(Tensor<T, 5, true, int> dists,
					   Tensor<int, 4, true, int> clusters,
					   Tensor<int, 1, true, int> sizes){
      // compute pairwise distances across time.
      
    }

    template <typename T>
    __global__ void update_clusters_kernel(Tensor<T, 5, true, int> dists,
					   Tensor<int, 4, true, int> clusters,
					   Tensor<int, 1, true, int> sizes){
      // compute pairwise distances across time.
      
    }



    template <typename T>
    void update_clusters(Tensor<T, 5, true, int>& dists,
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
      auto grid = dim3(numGrids);
      auto block = dim3(numThreads);
      if (init_update){
	init_clusters_kernel<T>
	  <<<grid,block,0,stream>>>(dists,clusters,sizes);
      }else{
	update_clusters_kernel<T>
	  <<<grid,block,0,stream>>>(dists,clusters,sizes);
      }

    }




    //
    // Template Inits
    //
    
    void update_clusters(Tensor<float, 5, true, int>& dists,
			 Tensor<int, 4, true, int>& clusters,
			 Tensor<int, 1, true, int>& sizes,
			 bool init_update, cudaStream_t stream){
      update_clusters<float>(dists,clusters,sizes,init_update,stream);
    }

    void update_clusters(Tensor<half, 5, true, int>& dists,
			 Tensor<int, 4, true, int>& clusters,
			 Tensor<int, 1, true, int>& sizes,
			 bool init_update, cudaStream_t stream){
      update_clusters<half>(dists,clusters,sizes,init_update,stream);
    }


  }
}
