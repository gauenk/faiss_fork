/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>

namespace faiss {
namespace gpu {
    void runNnfSimpleBlockSelect(
        Tensor<float, 3, true>& inVals,
        Tensor<int, 2, true>& inKeys,
        Tensor<float, 3, true>& outVals,
        Tensor<int, 4, true>& outKeys,
	float valMean, bool comp_with_out,int k,
        cudaStream_t stream);

}
}