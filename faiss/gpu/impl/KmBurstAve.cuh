
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void kmb_ave(Tensor<float, 5, true, int> centroids,
		 Tensor<float, 4, true, int> ave,
		 cudaStream_t stream);

    void kmb_ave(Tensor<half, 5, true, int> centroids,
		 Tensor<half, 4, true, int> ave,
		 cudaStream_t stream);
  }
}