
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void runKmBurstL2Norm(Tensor<float, 5, true>& centroids,
			  Tensor<float, 4, true>& ave,
			  Tensor<int, 5, true>& blocks,
			  Tensor<float, 3, true>& vals,
			  int patchsize,int nblocks,
			  bool normSquared,
			  cudaStream_t stream);

    void runKmBurstL2Norm(Tensor<half, 5, true>& centroids,
			  Tensor<half, 4, true>& ave,
			  Tensor<int, 5, true>& blocks,
			  Tensor<float, 3, true>& vals,
			  int patchsize,int nblocks,
			  bool normSquared,
			  cudaStream_t stream);

  } // namespace gpu
} // namespace faiss
