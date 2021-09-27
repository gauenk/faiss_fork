/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/BroadcastSumBurst.cuh>
#include <faiss/gpu/impl/BurstPatchDistance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/BurstNnfL2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>

#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>

namespace faiss {
  namespace gpu {


template <typename T>
void runBurstPatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 4, true>& burst,
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
	bool computeL2) {

    // Size of proposed image 
    auto nframes = burst.getSize(0);
    auto nftrs = burst.getSize(1);
    auto heightPad = burst.getSize(2);
    auto widthPad = burst.getSize(3);
    int nblocks_total = blockLabels.getSize(1);
    int nblocks2 = nblocks*nblocks;
    int pad = std::floor(patchsize/2) + std::floor(nblocks/2);
    int psHalf = std::floor(patchsize/2);
    constexpr int nstreams = 2;

    // Size of vals image
    auto height = outDistances.getSize(0);
    auto width = outDistances.getSize(1);
    auto kOut = outDistances.getSize(2);

    // Size of indices image
    auto nframes_outInd = outIndices.getSize(0);
    auto heightInd = outIndices.getSize(1);
    auto widthInd = outIndices.getSize(2);
    auto kOutInd = outIndices.getSize(3);
    auto two = outIndices.getSize(4);
    
    // Assert same size
    FAISS_ASSERT(nframes == nframes_outInd);
    FAISS_ASSERT(height == (heightPad-2*pad));
    FAISS_ASSERT(width == (widthPad-2*pad));
    FAISS_ASSERT(height == heightInd);
    FAISS_ASSERT(width == widthInd);
    FAISS_ASSERT(kOut == k);
    FAISS_ASSERT(kOutInd == k);
    FAISS_ASSERT(two == 2);
    // FAISS_ASSERT(nblocks_total == utils::pow(nblocks2,nframes-1));

    // init for comparison right now, to be removed.
    thrust::fill(thrust::cuda::par.on(stream),
		 outDistances.data(),
		 outDistances.end(),
		 Limits<float>::getMax());
    
    // If we're querying against a 0 sized set, just return empty results
    if (height == 0 || width == 0  || nftrs == 0) {
      thrust::fill(
    		   thrust::cuda::par.on(stream),
    		   outDistances.data(),
    		   outDistances.end(),
    		   Limits<float>::getMax());
      
      thrust::fill(
    		   thrust::cuda::par.on(stream),
    		   outIndices.data(),
    		   outIndices.end(),
    		   -1);
      return;
    }

    // By default, aim to use up to 512 MB of memory for the processing, with
    // both number of queries and number of centroids being at least 512.
    int tileHeight = 0; // batchsize across height
    int tileWidth = 0; // batchsize across width
    int tileBlocks = 0; // batchsize across blocks
    chooseImageTileSize(
		   height, // image height
    		   width, // image width
    		   nftrs, // num of features per pixel
		   patchsize, // patchsize
		   nblocks_total, // number of image blocks to search
    		   sizeof(T),
    		   res->getTempMemoryAvailableCurrentDevice(),
    		   tileHeight,
    		   tileWidth,
		   tileBlocks);
    int numHeightTiles = utils::divUp(height, tileHeight);
    int numWidthTiles = utils::divUp(width, tileWidth);
    int numBlockTiles = utils::divUp((nblocks_total), tileBlocks);

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Create index selection for BLOCKS to allow for Stream.

    DeviceTensor<int, 3, true> bl1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nframes,nblocks_total,2}); // revisit me
    DeviceTensor<int, 3, true> bl2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nframes,nblocks_total,2}); // revisit me
    bl1.copyFrom(blockLabels,stream);
    bl2.copyFrom(blockLabels,stream);
    DeviceTensor<int, 3, true>* blockLabelsList[2] = {&bl1,&bl2};

    // DeviceTensor<int, 3, true>* blockLabelsList[nstreams];
    // for (int i = 0; i < nstreams; ++i){
    //   DeviceTensor<int, 3, true> blockLabel_i(res,
    // 	makeTempAlloc(AllocType::Other, stream),
    // 	{nframes,nblocks_total,2}); // revisit me
    //   blockLabel_i.copyFrom(blockLabels,stream);
    //   blockLabelsList[i] = &blockLabel_i;
    // }

    //
    // Temporary memory space to *execute* a single batch
    //
    DeviceTensor<float, 3, true> distanceBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true>* distanceBufs[2] = {&distanceBuf_1,
						     &distanceBuf_2};

    // DeviceTensor<float, 3, true>* distanceBufs[nstreams];
    // for (int i = 0; i < nstreams; ++i){
    //   DeviceTensor<float, 3, true> distanceBuf_i(res,
    // 	makeTempAlloc(AllocType::Other, stream),
    // 	{tileHeight, tileWidth, tileBlocks});
    //   distanceBufs[i] = &distanceBuf_i;
    // }

    //
    // Temporary memory space to *ave* a single batch of images
    //

    int tileHeightPad = tileHeight + 2*pad;
    int tileWidthPad = tileWidth + 2*pad;
    DeviceTensor<T, 4, true> aveBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    // DeviceTensor<T, 4, true>* aveBufs[2] = {&aveBuf_1,&aveBuf_2};
    DeviceTensor<T, 4, true>* aveBufs[2];
    aveBufs[0] = &aveBuf_1;
    aveBufs[1] = &aveBuf_2;

    // DeviceTensor<T, 4, true>* aveBufs[2];
    // std::vector<DeviceTensor<T, 4, true>> aveBufsVec;
    // aveBufsVec.reserve(2);
// #pragma unroll
//     for (int i = 0; i < 2; ++i){ 
      // DeviceTensor<T, 4, true> aveBuf_i(res,
      // 	  makeTempAlloc(AllocType::Other, stream),
      // 	  {nftrs, tileBlocks, tileHeightPad, tileWidthPad});
      // aveBufs.push_back(aveBuf_i);
      // aveBufsVec.push_back(DeviceTensor<T, 4, true>(res,
      // 	  makeTempAlloc(AllocType::Other, stream),
      // 	{nftrs, tileBlocks, tileHeightPad, tileWidthPad}));
      // aveBufs[i] = &aveBuf_i;
    //   aveBufs[i] = new DeviceTensor<T, 4, true> (res,
    //   	  makeTempAlloc(AllocType::Other, stream),
    //   	  {nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    // }

    // DeviceTensor<T, 4, true>* aveBufs[nstreams] = {};
    // for (int i = 0; i < nstreams; ++i){
    //   auto aveBuf_i = DeviceTensor<T, 4, true>(res,
    // 	  makeTempAlloc(AllocType::Other, stream),
    // 	  {nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    //   res->syncDefaultStreamCurrentDevice();
    //   aveBufs[i] = &aveBuf_i;
    // }

    // Streams allow for concurrent kernel execs.
    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;
    bool interrupt = false;

    // Tile HEIGHT pixels
    for (int i = 0; i < height; i += tileHeight) {
        if (interrupt || InterruptCallback::is_interrupted()) {
            interrupt = true;
            break;
        }

	// create indices for height tiling
        int curHeightSize = std::min(tileHeight, height - i);

	int paddedWidthSize = tileWidth + 2*(pad);
	int paddedHeightSize = tileHeight + 2*(pad);
	paddedHeightSize = std::min(paddedHeightSize,heightPad - i);

	// create views from height tile 
        auto outDistanceHeightView = outDistances.narrow(0, i, curHeightSize);
        auto outIndexHeightView = outIndices.narrow(1, i, curHeightSize);
	auto burstHeightView = burst.narrow(2, i, paddedHeightSize);

	// Tile WIDTH pixels
        for (int j = 0; j < width; j += tileWidth) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }

	    // create indices for height tiling
            int curWidthSize = std::min(tileWidth, width - j);
	    int paddedWidthSize = tileWidth + 2*(pad);
	    paddedWidthSize = std::min(paddedWidthSize,widthPad - j);

	    // view from width tiling
            auto outDistanceView = outDistanceHeightView.narrow(1, j, curWidthSize);
            auto outIndexView = outIndexHeightView.narrow(2, j, curWidthSize);
            auto burstView = burstHeightView.narrow(3, j, paddedWidthSize);

	    for (int blk = 0; blk < nblocks_total; blk += tileBlocks) {
	      if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
	      }

	      // 
	      // View for Buffers 
	      //

	      auto curBlockSize = std::min(tileBlocks, nblocks_total - blk);
	      auto aveView = aveBufs[curStream]
	      	->narrow(1, 0, curBlockSize)
	      	.narrow(2, 0, curHeightSize+2*psHalf)
	      	.narrow(3,0, curWidthSize+2*psHalf);
	      auto distanceBufView = distanceBufs[curStream]
		->narrow(0, 0, curHeightSize)
		.narrow(1, 0, curWidthSize)
		.narrow(2,0,curBlockSize);

	      //
	      // View for Blocks
	      //

	      auto blockLabelView = blockLabelsList[curStream]
		->narrow(1, blk, curBlockSize);

	      //
	      // Compute Average 
	      //
	      runBurstAverage(burstView,blockLabelView,
	      		      aveView,patchsize,nblocks,
	      		      streams[curStream]);

	      //
	      // Execute Template Search
	      //
	      runBurstNnfL2Norm(burstView,
	      			aveView,
	      			blockLabelView,
	      			distanceBufView,
	      			// outDistanceView,
	      			patchsize,nblocks,true,
	      			streams[curStream]);

	      // select "topK" from "curBlockSize" of outDistances
	      // this "topK" selection is limited to a "curBlockSize" batch
	      runBurstNnfSimpleBlockSelect(distanceBufView,
	      				   blockLabelView,
	      				   outDistanceView,
	      				   outIndexView,
	      				   valMean,
	      				   false,k,streams[curStream]);


	    } // batching over blockTiles

	    /*******

		    Write Buffers to Output

		    1.) select the topK and store into the output

		    TODO:
		    - runBlockSelectPair

	    *******/
            // runNnfSimpleBlockSelect(outDistanceBufWidthView,
	    // 			    outIndexBufWidthView,
	    // 			    outDistanceView,
	    // 			    outIndexView, 0.,
	    // 			    true,k,streams[curStream]);

	    // std::cout << "updated streams." << std::endl;
            curStream = (curStream + 1) % 2;

        } // batching over widthTiles

    } // batching over heightTiles

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }
}


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
	bool computeL2){
  runBurstPatchDistance<float>(
        res,
        stream,
	burst,
	blockLabels,
        k,t,h,w,c,
	patchsize,
	nblocks,
	valMean,
        outDistances,
        outIndices,
	computeL2);
}

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
	bool computeL2){
  runBurstPatchDistance<half>(
        res,
        stream,
	burst,
	blockLabels,
        k,t,h,w,c,
	patchsize,
	nblocks,
	valMean,
        outDistances,
        outIndices,
	computeL2);

}



  } // end namespace gpu
} // end namespace faiss
