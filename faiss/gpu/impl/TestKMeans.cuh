
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_kmeans(int test_case,
		     Tensor<float, 5, true, int>& dists,
		     Tensor<float, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<float, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& cluster_sizes,
		     int patchsize, float offset, int kmeansK,
		     cudaStream_t stream);

    void test_kmeans(int test_case,
		     Tensor<half, 5, true, int>& dists,
		     Tensor<half, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<half, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& cluster_sizes,
		     int patchsize, float offset, int kmeansK,
		     cudaStream_t stream);

  }
}