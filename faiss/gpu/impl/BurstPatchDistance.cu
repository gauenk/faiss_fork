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
#include <faiss/gpu/utils/BurstBlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>
#include <faiss/gpu/utils/BlockIndices2Labels.cuh>

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
    constexpr int nstreams = 16;

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
    // thrust::fill(thrust::cuda::par.on(stream),
    // 		 outDistances.data(),
    // 		 outDistances.end(),
    // 		 Limits<float>::getMax());
    
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
    // tileBlocks = 128;
    int numHeightTiles = utils::divUp(height, tileHeight);
    int numWidthTiles = utils::divUp(width, tileWidth);
    int numBlockTiles = utils::divUp((nblocks_total), tileBlocks);
    // printf("(tileHeight,tileWidth,tileBlocks): (%d,%d,%d)\n",
    // 	   tileHeight,tileWidth,tileBlocks);

    

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    //
    // Temporary memory space to *execute* a single batch
    //
    DeviceTensor<float, 3, true> distanceBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_3(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_4(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_5(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_6(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_7(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_8(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_9(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_10(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_11(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_12(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_13(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_14(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_15(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true> distanceBuf_16(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, tileBlocks});
    DeviceTensor<float, 3, true>* distanceBufs[16] = {&distanceBuf_1,
    						      &distanceBuf_2,
    						      &distanceBuf_3,
    						      &distanceBuf_4,
    						      &distanceBuf_5,
    						      &distanceBuf_6,
    						      &distanceBuf_7,
    						      &distanceBuf_8,
    						      &distanceBuf_9,
    						      &distanceBuf_10,
    						      &distanceBuf_11,
    						      &distanceBuf_12,
    						      &distanceBuf_13,
    						      &distanceBuf_14,
    						      &distanceBuf_15,
    						      &distanceBuf_16};
    // std::vector<DeviceTensor<float, 3, true>> distanceBufs(nstreams,
    // 	  DeviceTensor<float, 3, true>(res,
    // 	  makeTempAlloc(AllocType::Other, stream),
    // 	  {tileHeight, tileWidth, tileBlocks}));
    // DeviceTensor<float, 3, true>** distanceBufs = new DeviceTensor<float, 3, true>*[nstreams];
    // std::vector<DeviceTensor<float, 3, true>> distanceBufs;
    // distanceBufs.resize(nstreams);
// #pragma unroll
//     for (int i = 0; i < nstreams; ++i){
//       auto distBuf_i = new DeviceTensor<float, 3, true>(res,
//     	makeTempAlloc(AllocType::Other, stream),
//     	{tileHeight, tileWidth, tileBlocks});
//       distanceBufs[i] = distBuf_i;
//       // distanceBufs.push_back(distBuf_i);
//     }

    // for (int i = 0; i < nstreams; ++i){
    //   for (int j = 0; j < distanceBufs[i].NumDim; ++j){
    // 	printf("[%d]: getSize(%d): %d\n",i,j,distanceBufs[i].getSize(j));
    //   }
    //   //isContiguous
    // }

    DeviceTensor<int, 3, true> indexingBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_3(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_4(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_5(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_6(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_7(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_8(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_9(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_10(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_11(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_12(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_13(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_14(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_15(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true> indexingBuf_16(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{tileHeight, tileWidth, k});
    DeviceTensor<int, 3, true>* indexingBufs[16] = {&indexingBuf_1,
						    &indexingBuf_2,
						    &indexingBuf_3,
						    &indexingBuf_4,
						    &indexingBuf_5,
						    &indexingBuf_6,
						    &indexingBuf_7,
						    &indexingBuf_8,
						    &indexingBuf_9,
						    &indexingBuf_10,
						    &indexingBuf_11,
						    &indexingBuf_12,
						    &indexingBuf_13,
						    &indexingBuf_14,
						    &indexingBuf_15,
						    &indexingBuf_16,};

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
    DeviceTensor<T, 4, true> aveBuf_3(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_4(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_5(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_6(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_7(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_8(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_9(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_10(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_11(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_12(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_13(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_14(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_15(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true> aveBuf_16(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeightPad, tileWidthPad});
    DeviceTensor<T, 4, true>* aveBufs[16];
    aveBufs[0] = &aveBuf_1;
    aveBufs[1] = &aveBuf_2;
    aveBufs[2] = &aveBuf_3;
    aveBufs[3] = &aveBuf_4;
    aveBufs[4] = &aveBuf_5;
    aveBufs[5] = &aveBuf_6;
    aveBufs[6] = &aveBuf_7;
    aveBufs[7] = &aveBuf_8;
    aveBufs[8] = &aveBuf_9;
    aveBufs[9] = &aveBuf_10;
    aveBufs[10] = &aveBuf_11;
    aveBufs[11] = &aveBuf_12;
    aveBufs[12] = &aveBuf_13;
    aveBufs[13] = &aveBuf_14;
    aveBufs[14] = &aveBuf_15;
    aveBufs[15] = &aveBuf_16;

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
	      // printf("(curHeightSize,curWidthSize,curBlockSize): (%d,%d,%d)\n",
	      // 	     curHeightSize,curWidthSize,curBlockSize);
	      auto aveView = aveBufs[curStream]
	      	->narrow(1, 0, curBlockSize)
	      	.narrow(2, 0, curHeightSize + 2*psHalf)
	      	.narrow(3,0, curWidthSize + 2*psHalf);
	      auto distanceBufView = distanceBufs[curStream]
		->narrow(0, 0, curHeightSize)
		.narrow(1, 0, curWidthSize)
		.narrow(2,0,curBlockSize);

	      //
	      // View for Blocks
	      //
	      auto blockLabelView = blockLabels.narrow(1, blk, curBlockSize);

	      //
	      // Compute Average 
	      //

	      runBurstAverage(burstView,blockLabelView,
	      		      aveView,patchsize,nblocks,
	      		      streams[curStream]);

	      //
	      // Execute Template Search
	      //

	      runBurstNnfL2Norm(burstView,aveView,
	      			blockLabelView,
	      			distanceBufView,
	      			// outDistanceView,
	      			patchsize,nblocks,true,
	      			streams[curStream]);

	      // 
	      //  Top K Selection 
	      //
	      // select "topK" from "curBlockSize" of outDistances
	      // this "topK" selection is limited to a "curBlockSize" batch
	      //

	      runBurstNnfSimpleBlockSelect(distanceBufView,
	      				   blockLabelView,
	      				   outDistanceView,
	      				   outIndexView,
	      				   valMean,
	      				   false,k,streams[curStream]);


	      // auto indexingBuf = indexingBufs[curStream]
	      // 	->narrow(0,0,curHeightSize)
	      // 	.narrow(1,0,curWidthSize);
	      // runBurstBlockSelect(distanceBufView,
	      // 			  // blockLabelView,
	      // 			  outDistanceView,
	      // 			  indexingBuf,
	      // 			  //outIndexView,
	      // 			  // valMean,
	      // 			  false,k,streams[curStream]);
	      // runBlockIndices2Labels(indexingBuf,
	      // 			     outIndexView,
	      // 			     blockLabelView,
	      // 			     streams[curStream]);
				     


	    } // batching over blockTiles

	    // 
	    //  Top K Selection: Compare across Inputs & Outputs (e.g. "Pairs")
	    //
	    // runBurstBlockSelectPairs(distanceBufView,
	    // 			     // blockLabelView,
	    // 			     outDistanceView,
	    // 			     indexingBuf,
	    // 			     //outIndexView,
	    // 			     // valMean,
	    // 			     false,k,streams[curStream]);

	    // 
	    //  Convert topK "BlockLabel INDICES" to "BlockLabel VALS"
	    //
	    // convertLocs2Blocks(indexBuf


            curStream = (curStream + 1) % nstreams;

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
