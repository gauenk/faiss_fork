
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    void kmb_ave4d(Tensor<float, 4, true, int> tensor,
		   Tensor<float, 3, true, int> ave,
		   cudaStream_t stream);
    
    void kmb_ave4d(Tensor<half, 4, true, int> tensor,
		   Tensor<half, 3, true, int> ave,
		   cudaStream_t stream);
  }
}