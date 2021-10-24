
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void pairwise_distances(Tensor<float, 5, true, int>& dists,
			    Tensor<float, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<float, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, cudaStream_t stream);

    void pairwise_distances(Tensor<half, 5, true, int>& dists,
			    Tensor<half, 4, true, int>& burst,
			    Tensor<int, 5, true, int>& blocks,
			    Tensor<half, 5, true, int>& centroids,
			    Tensor<int, 4, true, int>& clusters,
			    int patchsize, cudaStream_t stream);

  }
}