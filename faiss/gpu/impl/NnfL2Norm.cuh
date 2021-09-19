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

void runNnfL2Norm(
	Tensor<float, 3, true>& ref,
	Tensor<float, 3, true>& target,
        Tensor<int, 2, true>& blocks,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream);

void runNnfL2Norm(
	Tensor<half, 3, true>& ref,
	Tensor<half, 3, true>& target,
        Tensor<int, 2, true>& blocks,
        Tensor<float, 3, true>& vals,
	int patchsize,
	int nblocks,
        bool normSquared,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
