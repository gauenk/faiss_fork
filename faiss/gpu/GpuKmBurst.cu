/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuKmBurst.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmBurstDistance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
  namespace gpu {

    template <typename T>
    void bfKmBurstConvert(GpuResourcesProvider* prov,
			   const GpuKmBurstParams& args) {
      // Validate the input data
      // FAISS_THROW_IF_NOT_MSG(
      //         args.k > 0 || args.k == -1,
      //         "bfKmBurst: k must be > 0 for top-k reduction, "
      //         "or -1 for all pairwise distances");
      // FAISS_THROW_IF_NOT_MSG(args.dims > 0, "bfKmBurst: dims must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numVectors > 0, "bfKmBurst: numVectors must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.vectors, "bfKmBurst: vectors must be provided (passed null)");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.numQueries > 0, "bfKmBurst: numQueries must be > 0");
      // FAISS_THROW_IF_NOT_MSG(
      //         args.queries, "bfKmBurst: queries must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outDistances,
              "bfKmBurst: outDistances must be provided (passed null)");
      FAISS_THROW_IF_NOT_MSG(
              args.outIndices || args.k == -1,
              "bfKmBurst: outIndices must be provided (passed null)");

      // Don't let the resources go out of scope
      // std::cout << "about to get res" << std::endl;
      auto resImpl = prov->getResources();
      auto res = resImpl.get();
      // std::cout << "res" << std::endl;
      auto device = getCurrentDevice();
      auto stream = res->getDefaultStreamCurrentDevice();
      // std::cout << "Got the Stream!" << std::endl;

      int psHalf = args.ps/2;
      auto burst = toDeviceTemporary<T, 4>(res,device,
					   const_cast<T*>(reinterpret_cast<const T*>
							  (args.burst)),
					   stream,{args.c,args.t,
						     args.h+2*psHalf,args.w+2*psHalf});
      auto sranges = toDeviceTemporary<int, 5>(res,device,args.search_ranges,stream,
					       {2,args.t,args.nsearch,args.h,args.w});
      auto init_blocks = toDeviceTemporary<int, 3>(res,device,args.init_blocks,stream,
						   {args.t,args.h,args.w});
      auto outDists = toDeviceTemporary<float, 3>(res,device,args.outDistances,
						  stream,{args.k,args.h,args.w});

      if (args.outIndicesType == IndicesDataType::I64) {
        // The brute-force API only supports an interface for i32 indices only,
        // so we must create an output i32 buffer then convert back
        DeviceTensor<int, 5, true> outIntIndices(res,
						 makeTempAlloc(AllocType::Other, stream),
						 {2, args.t, args.k, args.h, args.w});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation

        bfKmBurstOnDevice<T>(
                res,
		device,
                stream,
		burst,
		sranges,
		init_blocks,
                args.kmeansK,
                args.k,
		args.t,
		args.h,
		args.w,
		args.c,
		args.ps,
		args.nsearch,
		args.std,
                args.metric,
                args.metricArg,
                outDists,
                outIntIndices,
                args.ignoreOutDistances);

        // Convert and copy int indices out
        auto tOutIndices = toDeviceTemporary<Index::idx_t, 5>(res,device,
							      (Index::idx_t*)
							      args.outIndices,
							      stream,
							      {2, args.t, args.k,
								 args.h, args.w});

        // Convert int to idx_t
        convertTensor<int, Index::idx_t, 5>(stream, outIntIndices, tOutIndices);

        // Copy back if necessary
        fromDevice<Index::idx_t, 5>(tOutIndices, (Index::idx_t*)args.outIndices, stream);

      } else if (args.outIndicesType == IndicesDataType::I32) {
        // We can use the brute-force API directly, as it takes i32 indices
        // FIXME: convert to int32_t everywhere?
        static_assert(sizeof(int) == 4, "");
        auto outIntIndices = toDeviceTemporary<int, 5>(res,device,
							(int*)args.outIndices,
							stream,
						       {2, args.t, args.k,
							  args.h, args.w});

        // Since we've guaranteed that all arguments are on device, call the
        // implementation
        bfKmBurstOnDevice<T>(
                res,
		device,
                stream,
		burst,
		sranges,
		init_blocks,
                args.kmeansK,
                args.k,
		args.t,
                args.h,
                args.w,
                args.c,
		args.ps,
		args.nsearch,
		args.std,
                args.metric,
                args.metricArg,
                outDists,
                outIntIndices,
                args.ignoreOutDistances);

        // Copy back if necessary
        fromDevice<int, 5>(outIntIndices, (int*)args.outIndices, stream);
      } else {
        FAISS_THROW_MSG("unknown outIndicesType");
      }

      // Copy distances back if necessary
      fromDevice<float, 3>(outDists, args.outDistances, stream);
    }

    void bfKmBurst(GpuResourcesProvider* res, const GpuKmBurstParams& args) {
      // For now, both vectors and queries must be of the same data type

      if (args.dType == DistanceDataType::F32) {
	bfKmBurstConvert<float>(res, args);
      } else if (args.dType == DistanceDataType::F16) {
      	bfKmBurstConvert<half>(res, args);
      } else {
        FAISS_THROW_MSG("unknown vectorType");
      }
    }


  }
}