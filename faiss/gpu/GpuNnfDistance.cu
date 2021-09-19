/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuNnfDistance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/ImagePatchDistance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
  namespace gpu {

    template <typename T>
    void bfNnfConvert(GpuResourcesProvider* prov, const GpuNnfDistanceParams& args) {
      // Validate the input data
      // FAISS_THROW_IF_NOT_MSG(
      //         args.k > 0 || args.k == -1,
      //         "bfNnf: k must be > 0 for top-k reduction, "
      //         "or -1 for all pairwise distances");
      // FAISS_THROW_IF_NOT_MSG(args.dims > 0, "bfNnf: dims must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numVectors > 0, "bfNnf: numVectors must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.vectors, "bfNnf: vectors must be provided (passed null)");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numQueries > 0, "bfNnf: numQueries must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.queries, "bfNnf: queries must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outDistances,
              "bfNnf: outDistances must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outIndices || args.k == -1,
              "bfNnf: outIndices must be provided (passed null)");

      // Don't let the resources go out of scope
      // std::cout << "about to get res" << std::endl;
      auto resImpl = prov->getResources();
      auto res = resImpl.get();
      // std::cout << "res" << std::endl;
      auto device = getCurrentDevice();
      auto stream = res->getDefaultStreamCurrentDevice();
      // std::cout << "Got the Stream!" << std::endl;

      int pad = (int)std::floor(args.ps/2);
      auto targetImg = toDeviceTemporary<T, 3>(
					       res,
					       device,
					       const_cast<T*>(reinterpret_cast<const T*>
							      (args.targetImg)),
					       stream,
					       {args.c,args.h+2*pad,args.w+2*pad});
      auto refImg = toDeviceTemporary<T, 3>(
					    res,
					    device,
					    const_cast<T*>(reinterpret_cast<const T*>
							   (args.refImg)),
					    stream,
					    {args.c,args.h+2*pad,args.w+2*pad});
      auto blockLabels = toDeviceTemporary<int, 2>(
					    res,
					    device,
					    args.blockLabels,
					    stream,
					    {args.nblocks*args.nblocks,2});

      DeviceTensor<float, 2, true> refPatchNorms;
      if (args.refPatchNorms) {
        refPatchNorms = toDeviceTemporary<float, 2>(
						    res,
						    device,
						    const_cast<float*>
						    (args.refPatchNorms),
						    stream,
						    {args.h,args.w});
      }

      auto tOutDistances = toDeviceTemporary<float, 3>(
						       res,
						       device,
						       args.outDistances,
						       stream,
						       {args.h,args.w,args.k});

      if (args.outIndicesType == IndicesDataTypeNnf::I64) {
        // The brute-force API only supports an interface for i32 indices only,
        // so we must create an output i32 buffer then convert back
        DeviceTensor<int, 4, true> tOutIntIndices(
						  res,
						  makeTempAlloc(AllocType::Other, stream),
						  {args.h, args.w, args.k, 2});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation

        bfNnfOnDevice<T>(
                res,
		device,
                stream,
		targetImg,
		refImg,
                args.refPatchNorms ? &refPatchNorms : nullptr,
		blockLabels,
                args.k,
		args.h,
		args.w,
		args.c,
		args.ps,
		args.nblocks,
		args.valMean,
                args.metric,
                args.metricArg,
                tOutDistances,
                tOutIntIndices,
                args.ignoreOutDistances);

        // Convert and copy int indices out
        auto tOutIndices = toDeviceTemporary<Index::idx_t, 4>(
							      res,
							      device,
							      (Index::idx_t*)args.outIndices,
							      stream,
							      {args.h, args.w, args.k, 2});

        // Convert int to idx_t
        convertTensor<int, Index::idx_t, 4>(
					    stream, tOutIntIndices, tOutIndices);

        // Copy back if necessary
        fromDevice<Index::idx_t, 4>(
				    tOutIndices, (Index::idx_t*)args.outIndices, stream);

      } else if (args.outIndicesType == IndicesDataTypeNnf::I32) {
        // We can use the brute-force API directly, as it takes i32 indices
        // FIXME: convert to int32_t everywhere?
        static_assert(sizeof(int) == 4, "");

        auto tOutIntIndices = toDeviceTemporary<int, 4>(
							res,
							device,
							(int*)args.outIndices,
							stream,
							{args.h, args.w, args.k, 2});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation
        bfNnfOnDevice<T>(
                res,
		device,
                stream,
		targetImg,
		refImg,
                args.refPatchNorms ? &refPatchNorms : nullptr,
		blockLabels,
                args.k,
                args.h,
                args.w,
                args.c,
		args.ps,
		args.nblocks,
		args.valMean,
                args.metric,
                args.metricArg,
                tOutDistances,
                tOutIntIndices,
                args.ignoreOutDistances);

        // Copy back if necessary
        fromDevice<int, 4>(tOutIntIndices, (int*)args.outIndices, stream);
      } else {
        FAISS_THROW_MSG("unknown outIndicesType");
      }

      // Copy distances back if necessary
      fromDevice<float, 3>(tOutDistances, args.outDistances, stream);
    }

    void bfNnf(GpuResourcesProvider* res, const GpuNnfDistanceParams& args) {
      // For now, both vectors and queries must be of the same data type

      if (args.dType == DistanceDataTypeNnf::F32) {
	bfNnfConvert<float>(res, args);
      } else if (args.dType == DistanceDataTypeNnf::F16) {
      	bfNnfConvert<half>(res, args);
      } else {
        FAISS_THROW_MSG("unknown vectorType");
      }
    }


  }
}