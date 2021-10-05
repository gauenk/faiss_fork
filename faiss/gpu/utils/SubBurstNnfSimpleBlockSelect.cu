/**
 * Copyright (c) Kent Gauen
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/SubBurstNnfSimpleBlockSelect.cuh>

/****
     Select "topK" from "blockTileSize" of inVals
 ****/

#define ABS(N) (((N)<0)?(-(N)):((N)))

namespace faiss {
  namespace gpu {

    __global__ void subBurstNnfBlockSelect(
	Tensor<float, 3, true> inVals, // h, w, nsearch
	Tensor<int, 5, true> inKeys, // nsearch, h, w, t, two
	Tensor<float, 3, true> outVals, // h, w, k
	Tensor<int, 5, true> outKeys, // t, h, w, k, two
	float valMean, int patchsize, int nblocks) {

      int row = threadIdx.x + blockDim.x * blockIdx.x;
      int col = threadIdx.y + blockDim.y * blockIdx.y;
      int rowIn = row + patchsize/2;
      int colIn = col + patchsize/2;
      int nframes = inKeys.getSize(3);
      int numOfComps = inKeys.getSize(0);
      bool legal_row = row < inVals.getSize(0);
      bool legal_col = col < inVals.getSize(1);
      int k = outVals.getSize(2);
      int kidx = 0;

      if ( legal_row && legal_col ) {

	float outVal_max = ABS(outVals[row][col][k-1] - valMean);
	float outVal_curr = outVal_max;
	for (int comp = 0; comp < numOfComps; ++comp){

	  float inVal_raw = inVals[row][col][comp];
	  float inVal = ABS(inVal_raw - valMean);

	  if (inVal < outVal_max){
	    kidx = k-1;
	    outVal_curr = outVal_max;
	    while( inVal < outVal_curr && kidx > 0){
	      kidx -= 1;
	      outVal_curr = outVals[row][col][kidx];
	      outVal_curr = ABS(outVal_curr - valMean);
	    }
	    if (kidx != 0){ kidx += 1; }
	    else if (inVal > outVal_curr){ kidx += 1; }
	    // printf("an assign!: %d,%f\n",kidx,inVal);

	    // shift values up
	    for (int sidx = k-1; sidx > kidx; --sidx){
	      outVals[row][col][sidx] = (float)outVals[row][col][sidx-1];
	      for (int fidx = 0; fidx < nframes; ++fidx){
		outKeys[fidx][row][col][sidx][0] = (int)
		  outKeys[fidx][row][col][sidx-1][0];
		outKeys[fidx][row][col][sidx][1] = (int)
		  outKeys[fidx][row][col][sidx-1][1];
	      }
	    }

	    // assign new values
	    outVals[row][col][kidx] = inVal_raw;
	    for (int fidx = 0; fidx < nframes; ++fidx){
	      outKeys[fidx][row][col][kidx][0] = (int)inKeys[comp][rowIn][colIn][fidx][0];
	      outKeys[fidx][row][col][kidx][1] = (int)inKeys[comp][rowIn][colIn][fidx][1];
	    }
	    outVal_max = ABS(outVals[row][col][k-1]-valMean);

	  }
	}
      }
    }
    
    void runSubBurstNnfSimpleBlockSelect(
	Tensor<float, 3, true>& inVals, // h,w,nsearch
	Tensor<int, 5, true>& inKeys, // nsearch,h,w,t,two
	Tensor<float, 3, true>& outVals, // h,w,k
	Tensor<int, 5, true>& outKeys, // t,h,w,k,two
	float valMean, bool comp_with_out,
	int k, int patchsize, int nblocks,
	cudaStream_t stream){

      // assert shapes 
      int psHalf = (patchsize/2);
      int pad = (nblocks/2) + (patchsize/2);
      FAISS_ASSERT(outVals.getSize(0) == outKeys.getSize(1)); // height
      FAISS_ASSERT(outVals.getSize(1) == outKeys.getSize(2)); // width
      FAISS_ASSERT(inVals.getSize(0) == outVals.getSize(0)); // nframes
      FAISS_ASSERT(inVals.getSize(1) == outVals.getSize(1)); // width
      FAISS_ASSERT(inVals.getSize(2) == inKeys.getSize(0)); // batched search space
      FAISS_ASSERT(outKeys.getSize(0) == inKeys.getSize(3)); // nframes
      FAISS_ASSERT(outKeys.getSize(1) == (inKeys.getSize(1)-2*psHalf)); // height
      FAISS_ASSERT(outKeys.getSize(2) == (inKeys.getSize(2)-2*psHalf)); // width
      FAISS_ASSERT(outKeys.getSize(4) == inKeys.getSize(4)); // two
      FAISS_ASSERT(outVals.getSize(2) == k);
      FAISS_ASSERT(outKeys.getSize(3) == k);
      
      // setup kernel launch
      // keep it simple; each (h,w) index gets a thread, _not_ a block
      // it is not as parallel as it could be. 
      // this will probably have horrible warp divergence too
      int maxThreads = (int) getMaxThreadsCurrentDevice();
      // std::cout << "maxThreads: " << maxThreads << std::endl;
      int sqrtThreads = 32;//utils::pow(maxThreads*1.0, .5);
      
      auto nBlocksH = utils::divUp(inVals.getSize(0),sqrtThreads);
      auto nBlocksW = utils::divUp(inVals.getSize(1),sqrtThreads);
      
      // printf("(nBlocksH,nBlocksW,sqrtThreads): (%d,%d,%d)\n",nBlocksH,nBlocksW,sqrtThreads);
      auto grid = dim3(nBlocksH,nBlocksW);
      auto block = dim3(sqrtThreads,sqrtThreads);

      // launch kernel
      subBurstNnfBlockSelect<<<grid, block, 0, stream>>>(inVals, inKeys,
							 outVals, outKeys,
							 valMean,
							 patchsize, nblocks);
      CUDA_TEST_ERROR();
    }
    
  }
}