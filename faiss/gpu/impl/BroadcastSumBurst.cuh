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

// ave[f,blk,x,y] = (1/T) \sum_{t=1}^T burst[t,f,x+block[t][blk][0],y+block[t][blk][1]]
void runBurstAverage(
	Tensor<float, 4, true>& burst,
        Tensor<int, 3, true>& blocks,
	Tensor<float, 4, true>& ave,
        cudaStream_t stream);

void runBurstAverage(
	Tensor<half, 4, true>& burst,
        Tensor<int, 3, true>& blocks,
	Tensor<half, 4, true>& ave,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
