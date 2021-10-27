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
#include <faiss/gpu/impl/MeshSearchSpace.cuh>
#include <faiss/gpu/impl/KmUtils.cuh>
#include <faiss/gpu/impl/KMeans.cuh>
#include <faiss/gpu/impl/KmBurstL2Norm.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/BurstNnfL2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/BurstBlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>
#include <faiss/gpu/utils/BlockIndices2Labels.cuh>
#include <faiss/gpu/impl/KmUtils.cuh>

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
void runKmBurstDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<T, 4, true>& burst,
        Tensor<int, 5, true>& search_ranges,
        Tensor<int, 3, true>& init_blocks,
        int kmeansK,
        int k,
        int t,
        int h,
        int w,
        int c,
	int patchsize,
	int nsearch,
	float std,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
	bool computeL2) {

    fprintf(stdout,"start of KmBurstDistance.\n");
    // some locals vars
    int psHalf = std::floor(patchsize/2);
    constexpr int nstreams = 1;

    // Size of proposed image 
    auto nftrs = burst.getSize(0);
    auto nframes = burst.getSize(1);
    // auto heightPad = burst.getSize(2);
    // auto widthPad = burst.getSize(3);
    auto height_b = burst.getSize(2);
    auto width_b = burst.getSize(3);

    // Size of search ranges 
    int two_sr = search_ranges.getSize(0);
    int nframes_sr = search_ranges.getSize(1);
    int nsearch_sr = search_ranges.getSize(2);
    int height_sr = search_ranges.getSize(3);
    int width_sr = search_ranges.getSize(4);

    // Size of vals image
    auto kOut = outDistances.getSize(0);
    auto height = outDistances.getSize(1);
    auto width = outDistances.getSize(2);

    // Size of indices image
    auto two_ind = outIndices.getSize(0);
    auto nframes_ind = outIndices.getSize(1);
    auto kOut_ind = outIndices.getSize(2);
    auto height_ind = outIndices.getSize(3);
    auto width_ind = outIndices.getSize(4);
    
    // Assert same size
    FAISS_ASSERT(nframes == nframes_ind);
    FAISS_ASSERT(nframes == nframes_sr);
    FAISS_ASSERT(nsearch == nsearch_sr);
    // FAISS_ASSERT(height == (heightPad-2*psHalf));
    // FAISS_ASSERT(width == (widthPad-2*psHalf));
    FAISS_ASSERT(height == height_b);
    FAISS_ASSERT(width == width_b);
    FAISS_ASSERT(height == height_ind);
    FAISS_ASSERT(width == width_ind);
    FAISS_ASSERT(height == height_sr);
    FAISS_ASSERT(width == width_sr);
    FAISS_ASSERT(k == kOut);
    FAISS_ASSERT(k == kOut_ind);
    FAISS_ASSERT(2 == two_ind);
    FAISS_ASSERT(2 == two_sr);
    fprintf(stdout,"post asserts from KmBurstDistance.\n");

    // Algorithm vars
    int niters = 1;//nframes/2;
    int nframes_search = 3;
    int ref = nframes/2;
    int nclusters = -1;
    float mode = 0;
    int nblocks = utils::pow(nsearch,nframes_search);
    DeviceTensor<int, 1, true> cluster_sizes(res,
    	makeTempAlloc(AllocType::Other, stream),{kmeansK});
    DeviceTensor<int, 2, true> search_frames(res,
	makeTempAlloc(AllocType::Other, stream),{niters,nframes_search});
    DeviceTensor<int, 3, true> curr_blocks(res,
	makeTempAlloc(AllocType::Other, stream),{nframes,height,width});
    thrust::copy(thrust::cuda::par.on(stream), init_blocks.data(),
		 init_blocks.end(), curr_blocks.data());
    thrust::fill(thrust::cuda::par.on(stream), search_frames.data(),
		 search_frames.end(),ref);

    // default: fill without ref
    // TODO: allow for random frames across "niters"
    for( int i = 0; i < niters; ++i){
      int s = 0;
      for( int t = 0; t < nframes; ++t){
	if (t == ref){ continue; }
	cudaMemcpy(search_frames[i][s].data(),&t,
		   sizeof(int),cudaMemcpyHostToDevice);
	s++;
	if (s >= nframes_search){ break; }
      }
    }

    // init for comparison right now, to be removed.
    // thrust::fill(thrust::cuda::par.on(stream),
    // 		 outDistances.data(),
    // 		 outDistances.end(),
    // 		 Limits<float>::getMax());
    
    // If we're querying against a 0 sized set, just return empty results
    if (height == 0 || width == 0  || nftrs == 0) {
      thrust::fill(thrust::cuda::par.on(stream),
    		   outDistances.data(),
    		   outDistances.end(),
    		   Limits<float>::getMax());
      
      thrust::fill(thrust::cuda::par.on(stream),
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
    // tileBlocks = 128;
    int numHeightTiles = utils::divUp(height, tileHeight);
    int numWidthTiles = utils::divUp(width, tileWidth);
    int numBlockTiles = utils::divUp(nblocks, tileBlocks);
    // printf("(tileHeight,tileWidth,tileBlocks): (%d,%d,%d)\n",
    // 	   tileHeight,tileWidth,tileBlocks);

    

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation
    fprintf(stdout,"post gpu check.\n");

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


    DeviceTensor<int, 5, true> blockBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_3(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_4(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_5(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_6(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_7(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_8(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_9(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_10(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_11(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_12(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_13(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_14(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_15(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true> blockBuf_16(res,
    	makeTempAlloc(AllocType::Other, stream),
	{2, nframes, nblocks, tileHeight, tileWidth});
    DeviceTensor<int, 5, true>* blockBufs[16] = {&blockBuf_1,&blockBuf_2,
						 &blockBuf_3,&blockBuf_4,
						 &blockBuf_5,&blockBuf_6,
						 &blockBuf_7,&blockBuf_8,
						 &blockBuf_9,&blockBuf_10,
						 &blockBuf_11,&blockBuf_12,
						 &blockBuf_13,&blockBuf_14,
						 &blockBuf_15,&blockBuf_16};

    //
    // Temporary memory space for "clustering" and "centroid" vars
    //
    DeviceTensor<T, 5, true> kmDistBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
	{nframes, nframes, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 5, true>* kmDistBufs[1] = {&kmDistBuf_1};

    DeviceTensor<int, 4, true> clusterBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
	{nframes, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<int, 4, true>* clusterBufs[1] = {&clusterBuf_1};

    DeviceTensor<T, 5, true> centroidBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
	{nftrs, kmeansK, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 5, true>* centroidBufs[1] = {&centroidBuf_1};

    //
    // Temporary memory space to *ave* a single batch of images
    //

    DeviceTensor<T, 4, true> aveBuf_1(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_2(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_3(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_4(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_5(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_6(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_7(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_8(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_9(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_10(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_11(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_12(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_13(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_14(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_15(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
    DeviceTensor<T, 4, true> aveBuf_16(res,
    	makeTempAlloc(AllocType::Other, stream),
    	{nftrs, tileBlocks, tileHeight, tileWidth});
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

	// create views from height tile 
	auto curBlocksHeightView = curr_blocks.narrow(1, i, curHeightSize);
        auto outDistanceHeightView = outDistances.narrow(1, i, curHeightSize);
        auto outIndexHeightView = outIndices.narrow(3, i, curHeightSize);
	auto srangesHeightView = search_ranges.narrow(3, i, curHeightSize);

	// Tile WIDTH pixels
        for (int j = 0; j < width; j += tileWidth) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }

	    // create indices for height tiling
            int curWidthSize = std::min(tileWidth, width - j);

	    // view from width tiling
	    auto curBlocksView = curBlocksHeightView.narrow(2, j, curWidthSize);
            auto outDistanceView = outDistanceHeightView.narrow(2, j, curWidthSize);
            auto outIndexView = outIndexHeightView.narrow(4, j, curWidthSize);
	    auto srangesView = srangesHeightView.narrow(4, j, curWidthSize);

	    // Iterate over a subset "niters" times
	    for (int iter = 0; iter < niters; iter += 1){
	      if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
	      }
	      auto blocks = blockBufs[curStream]
		->narrow(3, 0, curHeightSize).narrow(4, 0, curWidthSize);
	      fprintf(stdout,"about to create search space.\n");
	      create_search_space(srangesView,blocks,curBlocksView,
	      			  search_frames,iter,streams[curStream]);


	      // Tile the Search-Space
	      for (int blk = 0; blk < nblocks; blk += tileBlocks) {
		if (InterruptCallback::is_interrupted()) {
		  interrupt = true;
		  break;
		}
		

		// get batch of search space
        	auto curBlockSize = std::min(tileBlocks, nblocks - blk);

        	printf("(curHeightSize,curWidthSize,curBlockSize): (%d,%d,%d)\n",
        		     curHeightSize,curWidthSize,curBlockSize);

		// 
		//  Views of Tensors
		//
        	auto blockView = blocks.narrow(2, blk, curBlockSize);
        	auto aveView = aveBufs[curStream]
		  ->narrow(1, 0, curBlockSize)
		  .narrow(2, 0, curHeightSize+2*psHalf)
		  .narrow(3, 0, curWidthSize+2*psHalf);
        	auto distanceBufView = distanceBufs[curStream]
		  ->narrow(0, 0, curHeightSize)
		  .narrow(1, 0, curWidthSize)
		  .narrow(2, 0, curBlockSize);
		auto kmDistView = kmDistBufs[curStream]
		  ->narrow(2, 0, curBlockSize)
		  .narrow(3, 0, curHeightSize)
		  .narrow(4, 0, curWidthSize);
        	auto clusterView = clusterBufs[curStream]
		  ->narrow(1, 0, curBlockSize)
		  .narrow(2, 0, curHeightSize)
		  .narrow(3, 0, curWidthSize);
        	auto centroidView = centroidBufs[curStream]
		  ->narrow(2, 0, curBlockSize)
		  .narrow(3, 0, curHeightSize)
		  .narrow(4, 0, curWidthSize);

		//
        	// Assert Shapes
		//
        	// FAISS_ASSERT(aveView.getSize(0) == burstView.getSize(0));
        	// FAISS_ASSERT(aveView.getSize(2) == burstView.getSize(2));
        	// FAISS_ASSERT(aveView.getSize(3) == burstView.getSize(3));

        	//
        	// Compute Clusters using Patches
        	//

		kmeans_clustering(kmDistView,burst,blockView,
				  centroidView,clusterView,
				  cluster_sizes,patchsize,
				  kmeansK,(float)0.,streams[curStream]);

		//
		// Compute Mode
		//

		// mode = compute_mode(cluster_sizes,patchsize,std);

		//
		// Compute Average of Clusters
		//

        	// runKmBurstAverage(centroidView,blockView,
		// 		aveView,patchsize,nsearch,
		// 		streams[curStream]);
        
        	// thrust::fill(thrust::cuda::par.on(stream),
        	// 		 aveView.data(),
        	// 		 aveView.end(),
        	// 		 0.);
        
        
        	//
        	// L2Norm over Patches
        	//

        	runKmBurstL2Norm(centroidView,
				 aveView,
				 blockView,
				 distanceBufView,
				 // outDistanceView,
				 patchsize,nsearch,true,
				 streams[curStream]);
        
        	// 
        	//  Top K Selection 
        	//

        	// runKmBurstTopK(distanceBufView,
		// 			     blockLabelView,
		// 			     outDistanceView,
		// 			     outIndexView,
		// 			     mode,
		// 			     false,k,streams[curStream]);
        				     
        
        
	      } // batching over blockTiles

	    } // iterating over a subset of frames
	    curStream = (curStream + 1) % nstreams;

        } // batching over widthTiles

    } // batching over heightTiles

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }
}


void runKmBurstDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<float, 4, true>& burst,
        Tensor<int, 5, true>& search_ranges,
        Tensor<int, 3, true>& init_blocks,
        int kmeansK,
        int k,
        int t,
        int h,
        int w,
        int c,
	int patchsize,
	int nsearch,
	float std,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
	bool computeL2){
  runKmBurstDistance<float>(
        res,
        stream,
	burst,
	search_ranges,
	init_blocks,
        kmeansK,k,t,h,w,c,
	patchsize,
	nsearch,
	std,
        outDistances,
        outIndices,
	computeL2);
}

void runKmBurstDistance(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<half, 4, true>& burst,
        Tensor<int, 5, true>& search_ranges,
        Tensor<int, 3, true>& init_blocks,
	int kmeansK,
        int k,
        int t,
        int h,
        int w,
        int c,
	int patchsize,
	int nsearch,
	float std,
        Tensor<float, 3, true>& outDistances,
        Tensor<int, 5, true>& outIndices,
	bool computeL2){
  runKmBurstDistance<half>(
        res,
        stream,
	burst,
	search_ranges,
	init_blocks,
        kmeansK,k,t,h,w,c,
	patchsize,
	nsearch,
	std,
        outDistances,
        outIndices,
	computeL2);

}



  } // end namespace gpu
} // end namespace faiss
