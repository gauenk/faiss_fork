
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void update_centroids(Tensor<float, 5, true, int>& dists,
			  Tensor<float, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<uint8_t, 4, true, int>& sizes,
			  bool init_update, cudaStream_t stream);

    void update_centroids(Tensor<half, 5, true, int>& dists,
			  Tensor<half, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<half, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<uint8_t, 4, true, int>& sizes,
			  bool init_update, cudaStream_t stream);


  }
}
