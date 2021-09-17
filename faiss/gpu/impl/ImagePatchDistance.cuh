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

void runImagePatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 3, true>& targetImg,
        Tensor<float, 3, true>& refImg,
        Tensor<float, 2, true>* refPatchNorms,
        Tensor<int, 2, true> blockLabels,
        int k,
	int h,
	int w,
	int c,
	int patchsize,
	int nblocks,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
	bool computeL2);

void runImagePatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 3, true>& targetImg,
        Tensor<half, 3, true>& refImg,
        Tensor<float, 2, true>* refPatchNorms,
        Tensor<int, 2, true> blockLabels,
        int k,
	int h,
	int w,
	int c,
	int patchsize,
	int nblocks,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
	bool computeL2);

template <typename T>
void bfNnfOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 3, true>& targetImg,
        Tensor<T, 3, true>& refImg,
        Tensor<float, 2, true>* refPatchNorms,
        Tensor<int, 2, true> blockLabels,
        int k,
	int h,
	int w,
	int c,
	int ps,
	int nblocks,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
        bool ignoreOutDistances) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    if ((metric == faiss::MetricType::METRIC_L2) ||
        (metric == faiss::MetricType::METRIC_Lp && metricArg == 2)) {

      // runImagePatchDistance(static_cast<T>(1.0));
                // k,h,w,c,ps,
		// ~ignoreOutDistances);

        runImagePatchDistance(
                resources,
                stream,
		targetImg,
		refImg,
		refPatchNorms,
		blockLabels,
                k,h,w,c,ps,nblocks,
                outDistances,
                outIndices,
		~ignoreOutDistances);
    }else{
            FAISS_THROW_FMT("[ImagePatchDistance]: unimplemented metric type %d", metric);
    }
}


} /* namespace gpu */
} /* namespace faiss */