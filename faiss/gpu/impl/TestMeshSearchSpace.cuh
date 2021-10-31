
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {


    //
    // Template Decl
    //
    
    void test_mesh_search_space(int test_case,
				Tensor<int, 5, true, int>& blocks,
				Tensor<int, 3, true>& init_blocks,
				Tensor<int, 5, true>& search_ranges,
				Tensor<int, 2, true>& search_frames,
				cudaStream_t stream);

  }
}