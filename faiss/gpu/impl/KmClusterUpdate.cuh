
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void update_clusters(Tensor<float, 5, true, int>& dists,
			 Tensor<int, 4, true, int>& clusters,
			 Tensor<int, 1, true, int>& sizes,
			 bool init_update, cudaStream_t stream);

    void update_clusters(Tensor<half, 5, true, int>& dists,
			 Tensor<int, 4, true, int>& clusters,
			 Tensor<int, 1, true, int>& sizes,
			 bool init_update, cudaStream_t stream);

  }
}