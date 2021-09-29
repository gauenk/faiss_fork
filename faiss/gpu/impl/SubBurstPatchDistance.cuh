/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/GeneralDistance.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

void runSubBurstPatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 4, true>& burst,
        Tensor<float, 3, true>& subAve,
        Tensor<int, 5, true>& blockLabels,
        Tensor<bool, 4, true>& mask,
        int k,
        int t,
	int h,
	int w,
	int c,
	int patchsize,
	int nblocks,
	float valMean,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
	bool computeL2);

void runSubBurstPatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 4, true>& burst,
        Tensor<half, 3, true>& subAve,
        Tensor<int, 5, true>& blockLabels,
        Tensor<bool, 4, true>& mask,
        int k,
        int t,
	int h,
	int w,
	int c,
	int patchsize,
	int nblocks,
	float valMean,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
	bool computeL2);

template <typename T>
void bfSubBurstNnfOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& burst,
        Tensor<T, 3, true>& subAve,
        Tensor<int, 5, true>& blockLabels,
        Tensor<bool, 4, true>& mask,
        int k,
	int t,
	int h,
	int w,
	int c,
	int ps,
	int nblocks,
	float valMean,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
        bool ignoreOutDistances) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    if ((metric == faiss::MetricType::METRIC_L2) ||
        (metric == faiss::MetricType::METRIC_Lp && metricArg == 2)) {

      // runSubBurstPatchDistance(static_cast<T>(1.0));
                // k,h,w,c,ps,
		// ~ignoreOutDistances);

        runSubBurstPatchDistance(
                resources,
                stream,
		burst,
		subAve,
		blockLabels,
		mask,
                k,t,h,w,c,ps,
		nblocks,valMean,
                outDistances,
                outIndices,
		~ignoreOutDistances);
    }else{
            FAISS_THROW_FMT("[SubBurstPatchDistance]: unimplemented metric type %d", metric);
    }
}


} /* namespace gpu */
} /* namespace faiss */