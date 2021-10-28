
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_cluster_update(int test_case,
			     Tensor<float, 5, true, int>& dists,
			     Tensor<float, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<uint8_t, 4, true, int>& sizes,
			     float offset, cudaStream_t stream);


    void test_cluster_update(int test_case,
			     Tensor<half, 5, true, int>& dists,
			     Tensor<half, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<uint8_t, 4, true, int>& sizes,
			     float offset, cudaStream_t stream);

  }
}