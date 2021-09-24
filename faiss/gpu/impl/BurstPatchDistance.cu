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
    int nblocks2 = nblocks*nblocks;
    int pad = std::floor(patchsize/2) + std::floor(nblocks/2);
    int psHalf = std::floor(patchsize/2);

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
		   nblocks, // number of image blocks to search
    		   sizeof(T),
    		   res->getTempMemoryAvailableCurrentDevice(),
    		   tileHeight,
    		   tileWidth,
		   tileBlocks);
    int numHeightTiles = utils::divUp(height, tileHeight);
    int numWidthTiles = utils::divUp(width, tileWidth);
    int numBlockTiles = utils::divUp((nblocks*nblocks), tileBlocks);

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Create index selection for BLOCKS to allow for Stream.
    DeviceTensor<int, 3, true> blockLabel1(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nframes,nblocks*nblocks,2});
    DeviceTensor<int, 3, true> blockLabel2(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nframes,nblocks*nblocks,2});
    blockLabel1.copyFrom(blockLabels,stream);
    blockLabel2.copyFrom(blockLabels,stream);
    DeviceTensor<int, 3, true>* blockLabelsList[2] = {&blockLabel1,
						      &blockLabel2};

    //
    // Temporary memory space to *execute* a single batch
    //
    DeviceTensor<float, 3, true> distanceBuf1(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf2(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true>* distanceBufs[2] = {&distanceBuf1,
						     &distanceBuf2};

    //
    // Temporary memory space to *ave* a single batch of images
    //
    int tileHeightPad = tileHeight + 2*pad;
    int tileWidthPad = tileWidth + 2*pad;
    DeviceTensor<T, 4, true> aveBuf1(res,
    	makeTempAlloc(AllocType::Other, stream),
	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true>* aveBufs[2] = {&aveBuf1,&aveBuf2};

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
        int start_i = i;//std::max(i-(pad), 0);
	int extra_pad_h = 0;//std::max(pad-(i-start_i),0);
	int paddedHeightSize = tileHeight + 2*(pad);// + extra_pad_h;
	paddedHeightSize = std::min(paddedHeightSize,heightPad-start_i);

	// create views from height tile 
        auto outDistanceHeightView = outDistances.narrow(0, i, curHeightSize);
        auto outIndexHeightView = outIndices.narrow(1, i, curHeightSize);
	auto burstHeightView = burst.narrow(2, start_i, paddedHeightSize);

	// Tile WIDTH pixels
        for (int j = 0; j < width; j += tileWidth) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }

	    // create indices for height tiling
            int curWidthSize = std::min(tileWidth, width - j);
	    int start_j = j;
	    int extra_pad_w = 0;
	    int paddedWidthSize = tileWidth + 2*(pad);
	    paddedWidthSize = std::min(paddedWidthSize,widthPad-start_j);

	    // view from width tiling
            auto burstView = burstHeightView.narrow(3, start_j, paddedWidthSize);
            auto outDistanceView = outDistanceHeightView.narrow(1, j, curWidthSize);
            auto outIndexView = outIndexHeightView.narrow(2, j, curWidthSize);

	    for (int blk = 0; blk < nblocks2; blk += tileBlocks) {
	      if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
	      }

	      // 
	      // View for Buffers 
	      //

	      auto curBlockSize = std::min(tileBlocks, nblocks2 - blk);
	      auto distanceBufView = distanceBufs[curStream]
		->narrow(0, 0, curHeightSize)
		.narrow(1, 0, curWidthSize)
		.narrow(2,0,curBlockSize);
	      auto aveView = aveBufs[curStream]
		->narrow(1, 0, curBlockSize)
		.narrow(2, 0, paddedHeightSize)
		.narrow(3,0,paddedWidthSize);

	      //
	      // View for Blocks
	      //

	      auto blockLabelView = blockLabelsList[curStream]
		->narrow(1, blk, curBlockSize);


	      //
	      // Compute Average 
	      //
	      runBurstAverage(burstView,blockLabelView,
			      aveView,streams[curStream]);


	      //
	      // Execute Template Search
	      //
	      runBurstNnfL2Norm(burstView,
				aveView,
				blockLabelView,
				// distanceBufView,
				outDistanceView,
				patchsize,nblocks,true,
				streams[curStream]);

	      // select "topK" from "curBlockSize" of outDistances
	      // this "topK" selection is limited to a "curBlockSize" batch
	      // runBurstNnfSimpleBlockSelect(distanceBufView,
	      // 				   blockLabelView,
	      // 				   outDistanceView,
	      // 				   outIndexView,
	      // 				   valMean,
	      // 				   false,k,streams[curStream]);


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
