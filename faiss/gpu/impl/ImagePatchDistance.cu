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
#include <faiss/gpu/impl/ImagePatchDistance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/NnfL2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/NnfSimpleBlockSelect.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>

namespace faiss {
  namespace gpu {


template <typename T>
void runImagePatchDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 3, true>& targetImg,
        Tensor<T, 3, true>& refImg,
        Tensor<float, 2, true>* refPatchNorms,
        Tensor<int, 2, true> blockLabels,
        int k,
        int h,
        int w,
        int c,
	int patchsize,
	int nblocks,
	float valMean,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
	bool computeL2) {

    std::cout << "hey yo!." << std::endl;
    // Size of proposed image 
    auto nftrsRef = refImg.getSize(0);
    auto heightPadRef = refImg.getSize(1);
    auto widthPadRef = refImg.getSize(2);
    // auto npix = height * width;
    fprintf(stdout,"nftrs: %d | height: %d | width: %d\n",
	    nftrsRef,heightPadRef,widthPadRef);
    auto nftrs = refImg.getSize(0);
    int pad = std::floor(patchsize/2);

    // Size of reference image
    auto nftrsTgt = targetImg.getSize(0);
    auto heightPadTgt = targetImg.getSize(1);
    auto widthPadTgt = targetImg.getSize(2);
    fprintf(stdout,"height: %d | width: %d | nftrs: %d\n",
	    heightPadTgt,widthPadTgt,nftrsTgt);

    // Size of vals image
    auto height = outDistances.getSize(0);
    auto width = outDistances.getSize(1);
    auto kOut = outDistances.getSize(2);

    // Size of indices image
    auto heightInd = outIndices.getSize(0);
    auto widthInd = outIndices.getSize(1);
    auto kOutInd = outIndices.getSize(2);
    auto two = outIndices.getSize(3);
    
    // Assert same size
    FAISS_ASSERT(nftrsRef == nftrsTgt);
    FAISS_ASSERT(heightPadRef == heightPadTgt);
    FAISS_ASSERT(widthPadRef == widthPadTgt);
    FAISS_ASSERT(height == (heightPadRef-2*pad));
    FAISS_ASSERT(width == (widthPadRef-2*pad));
    FAISS_ASSERT(width == (widthPadRef-2*pad));
    FAISS_ASSERT(height == heightInd);
    FAISS_ASSERT(width == widthInd);
    FAISS_ASSERT(kOut == k);
    FAISS_ASSERT(kOutInd == k);
    FAISS_ASSERT(two == 2);
      
    // The dimensions of the vectors to consider // height, width, k,two
    FAISS_ASSERT(outDistances.getSize(0) == height);
    FAISS_ASSERT(outIndices.getSize(0) == height);
    FAISS_ASSERT(outDistances.getSize(1) == width);
    FAISS_ASSERT(outIndices.getSize(1) == width);
    FAISS_ASSERT(outDistances.getSize(2) == k); // height, width, k,two
    FAISS_ASSERT(outIndices.getSize(2) == k);
    FAISS_ASSERT(outIndices.getSize(3) == 2);

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
    int tileBlocks = 0; // batchsize across width
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
    int numBlockTiles = utils::divUp(nblocks*nblocks, tileBlocks);
    fprintf(stdout,"patchsize: %d | nblocks: %d | k: %d\n",patchsize,nblocks,k);
    fprintf(stdout,"tileHeight: %d | tileWidth: %d | tileBlocks: %d\n",
	    tileHeight,tileWidth,tileBlocks);
    fprintf(stdout,"numHeightTiles: %d | numWidthTiles: %d | numBlockTiles: %d\n",
	    numHeightTiles,numWidthTiles,numBlockTiles);

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Create index selection for BLOCKS to allow for Stream.
    DeviceTensor<int, 2, true> blockLabel1(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nblocks*nblocks,2});
    DeviceTensor<int, 2, true> blockLabel2(
    	res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nblocks*nblocks,2});
    blockLabel1.copyFrom(blockLabels,stream);
    blockLabel2.copyFrom(blockLabels,stream);
    DeviceTensor<int, 2, true>* blockLabelsList[2] = {&blockLabel1, &blockLabel2};

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
    DeviceTensor<float, 3, true>* distanceBufs[2] = {&distanceBuf1, &distanceBuf2};

    // Temporary memory to *accumulate* current topK output from kernel executions.
    //
    // In words:
    // for batches of (Height,Width) store topK from each "blockTile" batch
    //

    // DeviceTensor<float, 3, true> outDistanceBuf1(
    // 	   res,
    // 	   makeTempAlloc(AllocType::Other, stream),
    // 	   {tileHeight, tileWidth, numBlockTiles * k});
    // DeviceTensor<float, 3, true> outDistanceBuf2(
    //         res,
    //         makeTempAlloc(AllocType::Other, stream),
    // 	    {tileHeight, tileWidth, numBlockTiles * k});
    // DeviceTensor<float, 3, true>* outDistanceBufs[2] = {
    //         &outDistanceBuf1, &outDistanceBuf2};

    // DeviceTensor<int, 4, true> outIndexBuf1(
    //         res,
    //         makeTempAlloc(AllocType::Other, stream),
    //         {tileHeight, tileWidth, numBlockTiles * k, 2});
    // DeviceTensor<int, 4, true> outIndexBuf2(
    //         res,
    //         makeTempAlloc(AllocType::Other, stream),
    //         {tileHeight, tileWidth, numBlockTiles * k, 2});
    // DeviceTensor<int, 4, true>* outIndexBufs[2] = {
    //         &outIndexBuf1, &outIndexBuf2};


    // Streams allow for concurrent kernel execs.
    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;
    bool interrupt = false;

    // (old) Tile HEIGHT pixels
    // (new) Tile over HEIGHT and WIDTH pixels in REF image
    for (int i = 0; i < numHeightTiles; i += tileHeight) {
        if (interrupt || InterruptCallback::is_interrupted()) {
            interrupt = true;
            break;
        }

        int curHeightSize = std::min(tileHeight, height - i);

        auto outDistanceHeightView = outDistances.narrow(0, i, curHeightSize);
        auto outIndexHeightView = outIndices.narrow(0, i, curHeightSize);

	// for (int nidx = 0; nidx < outDistanceHeightView.NumDim; nidx += 1){
	//   std::cout << "get size " << outDistanceHeightView.getSize(nidx) << std::endl;
	// }

        int start_i = std::max(i-2*pad, 0);
        auto refImgHeightView = refImg.narrow(1, start_i, curHeightSize + 2*pad);
	auto targetImgHeightView = targetImg.narrow(1, start_i, curHeightSize+2*pad);

	/***
	    view for accumulation buffers
	 ***/
        // auto outDistanceBufHeightView =
        //         outDistanceBufs[curStream]->narrow(0, 0, curHeightSize);
        // auto outIndexBufHeightView =
        //         outIndexBufs[curStream]->narrow(0, 0, curHeightSize);

        // (old) Tile WIDTH pixels
        // (new) Tile over NBLOCKS associated with a REF pixel
        for (int j = 0; j < numWidthTiles; j += tileWidth) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }
            int curWidthSize = std::min(tileWidth, width - j);

	    /***
		Images (split view to batch across input pixels.)
	    ***/
	    int start_j = std::max(j-2*pad, 0);
            auto refImgView = refImgHeightView
	      .narrow(2, start_j, curWidthSize+2*pad);
            auto targetImgView = targetImgHeightView
	      .narrow(2, start_j, curWidthSize+2*pad);
	    /***
		Buffers (for accumulating topK results from block batches)
	    ***/
            // auto outDistanceBufWidthView =
	    //   outDistanceBufHeightView.narrow(1, 0, curWidthSize);
            // auto outIndexBufWidthView =
	    //   outIndexBufHeightView.narrow(1, 0,curWidthSize);

	    /***
		Outputs (vals and locs)
	    ***/
            auto outDistanceView =
	      outDistanceHeightView.narrow(1, j, curWidthSize);
            auto outIndexView =
	      outIndexHeightView.narrow(1, j, curWidthSize);

	    for (int nidx = 0; nidx < refImgView.NumDim; nidx += 1){
	      std::cout << " " << nidx << " get size " << refImgView.getSize(nidx) << std::endl;
	    }
	    for (int nidx = 0; nidx < targetImgView.NumDim; nidx += 1){
	      std::cout << " " << nidx << " get size " << targetImgView.getSize(nidx) << std::endl;
	    }

	    for (int j = 0; j < numBlockTiles; j += tileBlocks) {
	      if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
	      }


	      auto curBlockSize = std::min(tileBlocks, nblocks*nblocks - j);


	      // 
	      // View for Buffers 
	      //

	      auto distanceBufView = distanceBufs[curStream]
		->narrow(0, 0, curHeightSize)
		.narrow(1, 0, curWidthSize)
		.narrow(2,0,curBlockSize);
	      // auto outDistanceBufBlockView =
	      // 	outDistanceBufRowView.narrow(2, blockTileSize * k, k);
	      // auto outIndexBufBlockView =
	      // 	outIndexBufRowView.narrow(2, blockTileSize * k, k);

	      //
	      // View for Inputs
	      //
	      auto blockLabelView = blockLabelsList[curStream]->narrow(0,j, curBlockSize);

	      // exec kernel
	      runNnfL2Norm(refImgView,
			   targetImgView,
			   blockLabelView,
			   distanceBufView,
			   // outDistanceView,
			   patchsize,nblocks,true,
			   streams[curStream]);

	      // select "topK" from "curBlockSize" of outDistances
	      // this "topK" selection is limited to a "curBlockSize" batch
	      runNnfSimpleBlockSelect(distanceBufView,
	      			      blockLabelView,
	      			      outDistanceView,
	      			      outIndexView,
	      			      // outDistanceBufBlockView,
	      			      // outIndexBufBlockView,
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

            curStream = (curStream + 1) % 2;

        } // batching over widthTiles

    } // batching over heightTiles

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }
}


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
	float valMean,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
	bool computeL2){
  runImagePatchDistance<float>(
        res,
        stream,
        targetImg,
        refImg,
        refPatchNorms,
	blockLabels,
        k,h,w,c,
	patchsize,
	nblocks,valMean,
        outDistances,
        outIndices,
	computeL2);
}

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
	float valMean,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 4, true>& outIndices,
	bool computeL2){
  runImagePatchDistance<half>(
        res,
        stream,
        targetImg,
        refImg,
        refPatchNorms,
	blockLabels,
        k,h,w,c,patchsize,
	nblocks,
	valMean,
        outDistances,
        outIndices,
	computeL2);

}



  } // end namespace gpu
} // end namespace faiss
