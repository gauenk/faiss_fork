
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void kmb_topK(Tensor<float, 3, true, int> dists,
		  Tensor<int, 5, true, int> blocks,
		  Tensor<int, 5, true, int> outInds,
		  Tensor<float, 3, true, int> outDists,
		  Tensor<float, 3, true, int> modes,
		  cudaStream_t stream);

    void kmb_topK(Tensor<float, 3, true, int> dists,
		  Tensor<int, 5, true, int> blocks,
		  Tensor<int, 5, true, int> outInds,
		  Tensor<float, 3, true, int> outDists,
		  Tensor<half, 3, true, int> modes,
		  cudaStream_t stream);

  }
}