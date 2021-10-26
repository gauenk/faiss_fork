
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    // kmeans clustering
    void kmeans_clustering(Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K, float offset,
			   cudaStream_t stream);

    void kmeans_clustering(Tensor<half, 5, true, int>& dists,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<int, 1, true, int>& sizes,
			   int patchsize, int K, float offset,
			   cudaStream_t stream);
    
  }
}

