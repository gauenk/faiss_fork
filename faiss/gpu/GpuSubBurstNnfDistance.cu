/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuSubBurstNnfDistance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SubBurstPatchDistance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
  namespace gpu {

    template <typename T>
    void bfSubBurstNnfConvert(GpuResourcesProvider* prov,
			   const GpuSubBurstNnfDistanceParams& args) {
      // Validate the input data
      // FAISS_THROW_IF_NOT_MSG(
      //         args.k > 0 || args.k == -1,
      //         "bfSubBurstNnf: k must be > 0 for top-k reduction, "
      //         "or -1 for all pairwise distances");
      // FAISS_THROW_IF_NOT_MSG(args.dims > 0, "bfSubBurstNnf: dims must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numVectors > 0, "bfSubBurstNnf: numVectors must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.vectors, "bfSubBurstNnf: vectors must be provided (passed null)");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numQueries > 0, "bfSubBurstNnf: numQueries must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.queries, "bfSubBurstNnf: queries must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outDistances,
              "bfSubBurstNnf: outDistances must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outIndices || args.k == -1,
              "bfSubBurstNnf: outIndices must be provided (passed null)");

      // Don't let the resources go out of scope
      // std::cout << "about to get res" << std::endl;
      auto resImpl = prov->getResources();
      auto res = resImpl.get();
      // std::cout << "res" << std::endl;
      auto device = getCurrentDevice();
      auto stream = res->getDefaultStreamCurrentDevice();
      // std::cout << "Got the Stream!" << std::endl;

      int psHalf = std::floor(args.ps/2);
      int pad = std::floor(args.ps/2) + std::floor(args.nblocks/2);
      auto burst = toDeviceTemporary<T, 4>(res,device,
					   const_cast<T*>(reinterpret_cast<const T*>
							  (args.burst)),
					   stream,
					   {args.sub_t,args.c,args.h+2*pad,args.w+2*pad});
      auto subAve = toDeviceTemporary<T, 3>(res,device,
						const_cast<T*>(
						reinterpret_cast<const T*>
							       (args.subAve)),
						stream,
						{args.c,args.h+2*psHalf,args.w+2*psHalf});
      auto blockLabels = toDeviceTemporary<int, 5>(res,device,
					    args.blockLabels,
					    stream,
					    {args.nblocks_total,
					       args.h+2*psHalf,args.w+2*psHalf,
					       args.sub_t,2});
      auto mask = toDeviceTemporary<bool, 4>(res,device,
					     const_cast<bool*>(
                                             reinterpret_cast<const bool*>
							    (args.mask)),
					    stream,
					    {args.nblocks_total,
					       args.h+2*psHalf,args.w+2*psHalf,
					       args.sub_t});
      auto tOutDistances = toDeviceTemporary<float, 3>(
						       res,
						       device,
						       args.outDistances,
						       stream,
						       {args.h,args.w,args.k});

      if (args.outIndicesType == IndicesDataType::I64) {
        // The brute-force API only supports an interface for i32 indices only,
        // so we must create an output i32 buffer then convert back
        DeviceTensor<int, 5, true> tOutIntIndices(res,
						  makeTempAlloc(AllocType::Other, stream),
						  {args.sub_t, args.h, args.w, args.k, 2});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation

        bfSubBurstNnfOnDevice<T>(
                res,
		device,
                stream,
		burst,
		subAve,
		blockLabels,
		mask,
                args.k,
		args.t,
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
        auto tOutIndices = toDeviceTemporary<Index::idx_t, 5>(res,device,
							      (Index::idx_t*)
							      args.outIndices,
							      stream,
							      {args.sub_t, args.h, args.w,
								 args.k, 2});

        // Convert int to idx_t
        convertTensor<int, Index::idx_t, 5>(stream, tOutIntIndices, tOutIndices);

        // Copy back if necessary
        fromDevice<Index::idx_t, 5>(tOutIndices, (Index::idx_t*)args.outIndices, stream);

      } else if (args.outIndicesType == IndicesDataType::I32) {
        // We can use the brute-force API directly, as it takes i32 indices
        // FIXME: convert to int32_t everywhere?
        static_assert(sizeof(int) == 4, "");

        auto tOutIntIndices = toDeviceTemporary<int, 5>(res,device,
							(int*)args.outIndices,
							stream,
							{args.sub_t, args.h,
							 args.w, args.k, 2});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation
        bfSubBurstNnfOnDevice<T>(
                res,
		device,
                stream,
		burst,
		subAve,
		blockLabels,
		mask,
                args.k,
		args.t,
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
        fromDevice<int, 5>(tOutIntIndices, (int*)args.outIndices, stream);
      } else {
        FAISS_THROW_MSG("unknown outIndicesType");
      }

      // Copy distances back if necessary
      fromDevice<float, 3>(tOutDistances, args.outDistances, stream);
    }

    void bfSubBurstNnf(GpuResourcesProvider* res,
		       const GpuSubBurstNnfDistanceParams& args) {
      // For now, both vectors and queries must be of the same data type

      if (args.dType == DistanceDataType::F32) {
	bfSubBurstNnfConvert<float>(res, args);
      } else if (args.dType == DistanceDataType::F16) {
      	bfSubBurstNnfConvert<half>(res, args);
      } else {
        FAISS_THROW_MSG("unknown vectorType");
      }
    }


  }
}