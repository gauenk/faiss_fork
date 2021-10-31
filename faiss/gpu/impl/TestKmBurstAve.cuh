
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_kmburst_ave(int test_case,
			  Tensor<float, 5, true, int>& dists,
			  Tensor<float, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<float, 4, true, int>& ave,
			  Tensor<float, 4, true, int>& modes,
			  int patchsize, float offset,
			  cudaStream_t stream);

    void test_kmburst_ave(int test_case,
			  Tensor<half, 5, true, int>& dists,
			  Tensor<half, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<half, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<half, 4, true, int>& ave,
			  Tensor<half, 4, true, int>& modes,
			  int patchsize, float offset,
			  cudaStream_t stream);

  }
}