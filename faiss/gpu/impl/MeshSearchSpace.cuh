
#include <algorithm>

namespace faiss {
  namespace gpu {
    void create_search_space(Tensor<int, 5, true>& search_ranges,
			     Tensor<int, 5, true>& blocks,
			     Tensor<int, 3, true>& curr_blocks,
			     Tensor<int, 2, true>& search_frames,
			     int iter, cudaStream_t stream);
    
    void write_to_file(Tensor<int, 5, true>& tensor5d,
		       const char* fn);


  }
}