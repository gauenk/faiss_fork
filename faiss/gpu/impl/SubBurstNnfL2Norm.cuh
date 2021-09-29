/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runSubBurstNnfL2Norm(
        Tensor<float, 4, true>& burst,
        Tensor<float, 4, true>& ave,
        Tensor<int, 5, true>& blocks,
        Tensor<bool, 4, true>& mask,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream);

void runSubBurstNnfL2Norm(
	Tensor<half, 4, true>& burst,
        Tensor<half, 4, true>& ave,
        Tensor<int, 5, true>& blocks,
        Tensor<bool, 4, true>& mask,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
