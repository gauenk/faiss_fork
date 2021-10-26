
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_kmburst_topK(int test_case,
			   Tensor<float, 3, true, int>& dists,
			   Tensor<int, 5, true, int>& indices,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<float, 4, true, int>& ave,
			   Tensor<float, 1, true, int>& modes,
			   int patchsize, float offset,
			   cudaStream_t stream);

    void test_kmburst_topK(int test_case,
			   Tensor<float, 3, true, int>& dists,
			   Tensor<int, 5, true, int>& indices,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<int, 4, true, int>& clusters,
			   Tensor<half, 4, true, int>& ave,
			   Tensor<float, 1, true, int>& modes,
			   int patchsize, float offset,
			   cudaStream_t stream);

  }
}