
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void self_pairwise_distances(Tensor<float, 5, true, int>& dists,
				 Tensor<float, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 int patchsize, float offset,
				 cudaStream_t stream);

    void self_pairwise_distances(Tensor<half, 5, true, int>& dists,
				 Tensor<half, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 int patchsize, float offset,
				 cudaStream_t stream);

  }
}