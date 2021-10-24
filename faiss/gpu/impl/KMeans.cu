

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
    void kmeans_clustering(Tensor<T, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<T, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K,
			   cudaStream_t stream){

      // create distances buf 
      // DeviceTensor<T, 5, true> dists(res,
      // 	makeTempAlloc(AllocType::Other, stream),
      // 	{nframes, nframes, nblocks, height, width});
      DeviceTensor<T, 5, true> dists;


      // init kmeans
      init_kmeans(dists,burst,blocks,centroids,clusters,sizes,stream);

      // cluster for iters
      int niters = 10;
      for (int i = 0; i < niters; ++i){

	// compute distances 
	compute_distances(dists,burst,blocks,centroids,clusters,patchsize,stream);

	// update kmeans assignments
	update_kmeans_state(dists,burst,blocks,centroids,clusters,
			    sizes,patchsize,stream);

      }
      
    }

    template <typename T>
    void compute_distances(Tensor<T, 5, true, int>& dists,
			   Tensor<T, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<T, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize, cudaStream_t stream){

      /*
	dists[t0,t1] = (bursts[t0,blocks[t0]] - bursts[t1,blocks[t1]])**2
      */
      pairwise_distances(dists,burst,blocks,centroids,clusters,
			 patchsize, stream);
    }

    template <typename T>
    void update_kmeans_state(Tensor<T, 5, true, int>& dists,
			     Tensor<T, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<T, 5, true, int>& centroids,
			     Tensor<int, 4, true, int>& clusters,
			     Tensor<int, 1, true, int>& sizes,
			     int patchsize, cudaStream_t stream){
      // are we init?
      constexpr bool init_update = false;

      // update cluster assignments; argmin over "t"
      update_clusters(dists,clusters,sizes,init_update,stream);

      // update centroids (means) given the cluster assignments; ave over subsets
      update_centroids(dists,burst,blocks,centroids,clusters,sizes,init_update,stream);

    }

    template <typename T>
    void init_kmeans(Tensor<T, 5, true, int>& dists,
		     Tensor<T, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<T, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& sizes,
		     cudaStream_t stream){

      // are we init?
      constexpr bool init_update = true;

      // set clusters to be (1,...,K,1,..,K,1,...,) repeating
      // update cluster assignments; argmin over "t"
      update_clusters(dists,clusters,sizes,init_update,stream);

      // update centroids (means) given the initial cluster assignments
      update_centroids(dists,burst,blocks,centroids,clusters,sizes,init_update,stream);

    }


    // ------------------------------------------------------
    //
    //         Initialize Templates
    //
    // ------------------------------------------------------

    void compute_distances(Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize, cudaStream_t stream){
      compute_distances<float>(dists,burst,blocks,centroids,clusters,patchsize,stream);
    }

    void compute_distances(Tensor<half, 5, true, int>& dists,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize, cudaStream_t stream){
      compute_distances<half>(dists,burst,blocks,centroids,clusters,patchsize,stream);
    }

    void update_kmeans_state(Tensor<float, 5, true, int>& dists,
			     Tensor<float, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<float, 5, true, int>& centroids,
			     Tensor<int, 4, true, int>& clusters,
			     Tensor<int, 1, true, int>& sizes,
			     int patchsize, cudaStream_t stream){
      update_kmeans_state<float>(dists,burst,blocks,centroids,
				 clusters,sizes,patchsize,stream);
    }

    void update_kmeans_state(Tensor<half, 5, true, int>& dists,
			     Tensor<half, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<half, 5, true, int>& centroids,
			     Tensor<int, 4, true, int>& clusters,
			     Tensor<int, 1, true, int>& sizes,
			     int patchsize, cudaStream_t stream){
      update_kmeans_state<half>(dists,burst,blocks,centroids,
				clusters,sizes,patchsize,stream);
    }

    void init_kmeans(Tensor<float, 5, true, int>& dists,
		     Tensor<float, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<float, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& sizes,
		     cudaStream_t stream){
      init_kmeans<float>(dists,burst,blocks,centroids,
			 clusters,sizes,stream);
    }

    void init_kmeans(Tensor<half, 5, true, int>& dists,
		     Tensor<half, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<half, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& sizes,
		     cudaStream_t stream){
      init_kmeans<half>(dists,burst,blocks,centroids,
			clusters,sizes,stream);
    }


    void kmeans_clustering(Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K,
			   cudaStream_t stream){
      kmeans_clustering<float>(burst,blocks,centroids,
			      clusters,sizes,
			      patchsize, K, stream);
    }

    void kmeans_clustering(Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K,
			   cudaStream_t stream){
      kmeans_clustering<half>(burst,blocks,centroids,
			      clusters,sizes,
			      patchsize, K, stream);
    }

  }
}
