
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_pairwise_distances(int test_case,
				 Tensor<float, 5, true, int>& dists,
				 Tensor<float, 5, true, int>& self_dists,
				 Tensor<float, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 Tensor<float, 5, true, int>& centroids,
				 int patchsize, float offset,
				 cudaStream_t stream);

    void test_pairwise_distances(int test_case,
				 Tensor<half, 5, true, int>& dists,
				 Tensor<half, 5, true, int>& self_dists,
				 Tensor<half, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 Tensor<half, 5, true, int>& centroids,
				 int patchsize, float offset,
				 cudaStream_t stream);

  }
}