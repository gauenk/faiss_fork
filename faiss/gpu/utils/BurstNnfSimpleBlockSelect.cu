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
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>

/****
     Select "topK" from "blockTileSize" of inVals
 ****/

#define ABS(N) ((N<0)?(-N):(N))

namespace faiss {
  namespace gpu {

    __global__ void burstNnfBlockSelect(
	Tensor<float, 3, true> inVals,
	Tensor<int, 3, true> inKeys,
	Tensor<float, 3, true> outVals,
	Tensor<int, 5, true> outKeys,
	float valMean) {

      int row = threadIdx.x + blockDim.x * blockIdx.x;
      int col = threadIdx.y + blockDim.y * blockIdx.y;
      int nframes = inKeys.getSize(0);
      int numOfComps = inKeys.getSize(1);
      bool legal_row = row < inVals.getSize(0);
      bool legal_col = col < inVals.getSize(1);
      int k = outVals.getSize(2);
      int kidx = 0;

      if ( legal_row && legal_col ) {

	float outVal_max = outVals[row][col][k-1]; // already corrected value
	float outVal_curr = outVal_max;
	for (int comp = 0; comp < numOfComps; ++comp){

	  float inVal = ABS(inVals[row][col][comp] - valMean);

	  if (inVal < outVal_max){
	    kidx = k-1;
	    outVal_curr = outVal_max;
	    while( inVal < outVal_curr && kidx > 0){
	      kidx -= 1;
	      outVal_curr = outVals[row][col][kidx];
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
	    outVals[row][col][kidx] = inVal;
	    for (int fidx = 0; fidx < nframes; ++fidx){
	      outKeys[fidx][row][col][kidx][0] = inKeys[fidx][comp][0];
	      outKeys[fidx][row][col][kidx][1] = inKeys[fidx][comp][1];
	    }
	    outVal_max = outVals[row][col][k-1];

	  }
	}
      }
    }
    
    void runBurstNnfSimpleBlockSelect(
	Tensor<float, 3, true>& inVals,
	Tensor<int, 3, true>& inKeys,
	Tensor<float, 3, true>& outVals,
	Tensor<int, 5, true>& outKeys,
	float valMean, bool comp_with_out,int k,
	cudaStream_t stream){

      // assert shapes 
      FAISS_ASSERT(outVals.getSize(0) == outKeys.getSize(1)); // height
      FAISS_ASSERT(outVals.getSize(1) == outKeys.getSize(2)); // width
      FAISS_ASSERT(inVals.getSize(0) == outVals.getSize(0)); // nframes
      FAISS_ASSERT(inVals.getSize(1) == outVals.getSize(1)); // width
      FAISS_ASSERT(inVals.getSize(2) == inKeys.getSize(1)); // batched search space
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
      burstNnfBlockSelect<<<grid, block, 0, stream>>>(inVals, inKeys,
						      outVals, outKeys,
						      valMean);
      CUDA_TEST_ERROR();
    }
    
  }
}