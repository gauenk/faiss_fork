/// convert block indices to labels

#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runBlockIndices2Labels(
        Tensor<int, 3, true>& inIndices,
	Tensor<int, 5, true>& outLabels,
        Tensor<int, 3, true>& blocks,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
