
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    // init kmeans 
    void init_kmeans(Tensor<float, 5, true, int>& dists,
		     Tensor<float, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<float, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& sizes,
		     cudaStream_t stream);
    void init_kmeans(Tensor<half, 5, true, int>& dists,
		     Tensor<half, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<half, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& sizes,
		     cudaStream_t stream);

    // update centroids and clusters
    void update_kmeans_state(Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize);
    void update_kmeans_state(Tensor<half, 5, true, int>& dists,
			     Tensor<half, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<half, 5, true, int>& centroids,
			     Tensor<int, 4, true, int>& clusters,
			     int patchsize);

    // compute distances 
    void compute_distances(Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize);
    void compute_distances(Tensor<half, 5, true, int>& dists,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   int patchsize);

    // kmeans clustering
    void kmeans_clustering(Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K,
			   cudaStream_t stream);

    void kmeans_clustering(Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K,
			   cudaStream_t stream);
    
  }
}

