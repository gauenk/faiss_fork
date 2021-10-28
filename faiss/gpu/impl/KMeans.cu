
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KMeans.cuh>
#include <faiss/gpu/impl/PairwiseDistances.cuh>
#include <faiss/gpu/impl/KmClusterUpdate.cuh>
#include <faiss/gpu/impl/KmCentroidUpdate.cuh>
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
    void kmeans_clustering(Tensor<T, 5, true, int>& dists,
			   Tensor<T, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<T, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<uint8_t, 4, true, int>& sizes,
			   int patchsize, int K, float offset,
			   cudaStream_t stream){

      /*************************************
      
             init kmeans
      
      *************************************/

      // set clusters to be (1,...,K,1,..,K,1,...,) repeating
      // update cluster assignments; argmin over "t"
      bool init_update = true;
      update_clusters(dists,burst,clusters,sizes,init_update,stream);

      // update centroids (means) given the initial cluster assignments
      update_centroids(dists,burst,blocks,centroids,clusters,
		       sizes,init_update,stream);
      

      /*************************************
      
             cluster for iters
      
      *************************************/

      init_update = false;
      int niters = 10;
      for (int i = 0; i < niters; ++i){

	// compute distances 
	pairwise_distances(dists,burst,blocks,centroids,patchsize,offset,stream);

	// update cluster assignments; argmin over "t"
	update_clusters(dists,burst,clusters,sizes,init_update,stream);

	// update centroids (means) given the cluster assignments; ave over subsets
	update_centroids(dists,burst,blocks,centroids,clusters,
			 sizes,init_update,stream);

      }
      
    }
    

    // ------------------------------------------------------
    //
    //         Initialize Templates
    //
    // ------------------------------------------------------

    void kmeans_clustering(Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<uint8_t, 4, true, int>& sizes,
			   int patchsize, int K, float offset,
			   cudaStream_t stream){
      kmeans_clustering<float>(dists,burst,blocks,centroids,
			       clusters,sizes, patchsize,
			       K, offset, stream);
    }

    void kmeans_clustering(Tensor<half, 5, true, int>& dists,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<uint8_t, 4, true, int>& sizes,
			   int patchsize, int K, float offset,
			   cudaStream_t stream){
      kmeans_clustering<half>(dists,burst,blocks,centroids,
			      clusters,sizes,patchsize,
			      K, offset, stream);
    }

  }
}
