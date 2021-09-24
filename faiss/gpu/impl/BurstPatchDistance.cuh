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

void runBurstPatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 4, true>& burst,
        Tensor<int, 3, true>& blockLabels,
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

void runBurstPatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 4, true>& burst,
        Tensor<int, 3, true>& blockLabels,
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
void bfBurstNnfOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& burst,
        Tensor<int, 3, true>& blockLabels,
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

      // runBurstPatchDistance(static_cast<T>(1.0));
                // k,h,w,c,ps,
		// ~ignoreOutDistances);

        runBurstPatchDistance(
                resources,
                stream,
		burst,
		blockLabels,
                k,t,h,w,c,ps,
		nblocks,valMean,
                outDistances,
                outIndices,
		~ignoreOutDistances);
    }else{
            FAISS_THROW_FMT("[BurstPatchDistance]: unimplemented metric type %d", metric);
    }
}


} /* namespace gpu */
} /* namespace faiss */