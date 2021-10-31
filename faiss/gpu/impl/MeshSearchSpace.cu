/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/NnfL2Norm.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <faiss/gpu/impl/MeshSearchSpace.cuh>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace faiss {
namespace gpu {

  template <int BlockTileSize>
  __global__ void create_search_space_kernel(Tensor<int, 5, true> sranges,
					     Tensor<int, 5, true> blocks,
					     Tensor<int, 3, true> curr_blocks,
					     Tensor<int, 2, true> search_frames,
					     int iter, bool NormLoop){

    // get CUDA indices
    int bidx = threadIdx.x + blockIdx.x * blockDim.x;

    // setup vars
    int nframes = sranges.getSize(1);
    int nblocks = blocks.getSize(2);
    int nsearch = sranges.getSize(2);
    int nfsearch = search_frames.getSize(1);
    bool is_ref = false;
    bool is_search = false;
    int div,sidx,sframe;
    int num_frames_searched;
    int ref = nframes/2;
    
    // run for height, width, and all frames
    for (int hidx = 0; hidx < sranges.getSize(3); ++hidx){
      for (int widx = 0; widx < sranges.getSize(4); ++widx){
	num_frames_searched = 0;
	for(int t = 0; t < nframes; ++t){

	  // check if "t in search_frames[iter]" ?
	  is_search = false;
	  for(int t_p = 0; t_p < nfsearch; ++t_p){
	    sframe = search_frames[iter][t_p];
	    if (sframe == t){
	      is_search = true;
	    }
	  }

	  // check if ref frame
	  is_ref = t == ref;

	  // we don't search the reference
	  // is_search = is_search && !is_ref;
	  if(is_search && is_ref){ is_search = false; }
	  
	  // get index from search range
	  if (is_search){
	    if (num_frames_searched == 0){
	      div = 1;
	    }else{
	      div = utils::pow(nsearch,num_frames_searched);
	    }
	    num_frames_searched++;
	    sidx = (bidx / div) % nsearch;
	  }else{
	    sidx = curr_blocks[t][hidx][widx];
	  }

	  // copy ranges to mesh
	  blocks[0][t][bidx][hidx][widx] = (int)sranges[0][t][sidx][hidx][widx];
	  blocks[1][t][bidx][hidx][widx] = (int)sranges[1][t][sidx][hidx][widx];
	  // blocks[1][t][bidx][hidx][widx] = is_search;
	  // blocks[0][t][bidx][hidx][widx] = t;
	  // blocks[1][t][bidx][hidx][widx] = sidx;

	}
      }
    }

  }

  void create_search_space(Tensor<int, 5, true>& search_ranges,
			   Tensor<int, 5, true>& blocks,
			   Tensor<int, 3, true>& curr_blocks,
			   Tensor<int, 2, true>& search_frames,
			   int iter, cudaStream_t stream){

    // kernel params
    int maxThreads = (int)getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 1;
    constexpr int colTileSize = 1;
    constexpr int blockTileSize = 1;

    // ASSERT SIZES
    FAISS_ASSERT(search_ranges.getSize(0) == blocks.getSize(0)); // two
    FAISS_ASSERT(search_ranges.getSize(1) == blocks.getSize(1)); // nframes
    FAISS_ASSERT(search_ranges.getSize(3) == blocks.getSize(3)); // height
    FAISS_ASSERT(search_ranges.getSize(4) == blocks.getSize(4)); // width
    FAISS_ASSERT(curr_blocks.getSize(0) == blocks.getSize(1)); // nframes
    FAISS_ASSERT(curr_blocks.getSize(1) == blocks.getSize(3)); // height
    FAISS_ASSERT(curr_blocks.getSize(2) == blocks.getSize(4)); // width
    
    // get sizes
    int nframes = blocks.getSize(1);
    int nblocks = blocks.getSize(2);
    int height = blocks.getSize(3);
    int width = blocks.getSize(4);
    int nsearch_frames = search_frames.getSize(1);
    int nsearch = search_ranges.getSize(2);

    // Threads
    int dim = nblocks;
    bool normLoop = false;//dim > maxThreads;
    int numThreads = std::min(dim, maxThreads);

    // Grids
    int numBlockBlocks = utils::divUp(nblocks, maxThreads);

    // Launch
    auto grid = dim3(numBlockBlocks);
    auto block = dim3(numThreads);
    create_search_space_kernel<blockTileSize>
      <<<grid, block, 0, stream>>>(search_ranges, blocks,
    				   curr_blocks, search_frames, 
    				   iter, normLoop);
    CUDA_TEST_ERROR();


    // write to file for testing
    // cudaDeviceSynchronize();
    // const char sranges_fn[50] = "search_ranges.txt";
    // write_to_file(search_ranges,sranges_fn);
    // const char block_fn[50] = "blocks.txt";
    // write_to_file(blocks,block_fn);
    
  }

  void write_to_file(Tensor<int, 5, true>& tensor5d,const char* fn){
    int* num = (int*)malloc(sizeof(int));
    FILE *file;
    file = std::fopen(fn,"w");
    for (int i1 = 0; i1 < tensor5d.getSize(1); ++i1){
      for (int i2 = 0; i2 < tensor5d.getSize(2); ++i2){
	for (int i3 = 0; i3 < tensor5d.getSize(3); ++i3){
	  for (int i4 = 0; i4 < tensor5d.getSize(4); ++i4){
	    for (int i0 = 0; i0 < tensor5d.getSize(0); ++i0){
	      // num = (int)*(tensor5d[i0][i1][i2][i3][i4].data());
	      // num = (int*)(tensor5d[0][0][0][0][0].data());
	      cudaMemcpy(num,tensor5d[i0][i1][i2][i3][i4].data(),
			 sizeof(int),cudaMemcpyDeviceToHost);
	      // num = (int)*(tensor5d[i0][i1][i2][i3][i4].data());
	      // num = tensor5d.data();
	      // std::cout << *num << std::endl;
	      fprintf(file,"%d\n",*num);
	    }
	    // fprintf(file,"\n");
	  }
	}
      }
    }
    std::fclose(file);
  }

}
}