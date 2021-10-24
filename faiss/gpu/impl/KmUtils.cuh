/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/gpu/utils/Tensor.cuh>

//
// Shared utilities for brute-force distance calculations
//

namespace faiss {
  namespace gpu {


    // void create_search_space(Tensor<int, 5, true>& search_ranges,
    // 			     Tensor<int, 5, true>& blocks,
    // 			     Tensor<int, 2, true>& search_frames,
    // 			     int iter, int start_h, int end_h,
    // 			     int start_w, int end_w, cudaStream_t stream){
    //   fprintf(stdout,"create search space.\n");
    // }

    float compute_mode(Tensor<int, 1, true>& cluster_sizes,
		       int patchsize, float std){
      return 0;
    }

  }
}