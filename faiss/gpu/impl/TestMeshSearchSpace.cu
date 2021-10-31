
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/MeshSearchSpace.cuh>
#include <faiss/gpu/impl/TestMeshSearchSpace.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>


namespace faiss {
  namespace gpu {

    namespace test_mss{
      //
      // Test cases
      //

      void test_case_0(Tensor<int, 5, true, int>& blocks,
		       Tensor<int, 3, true>& init_blocks,
		       Tensor<int, 5, true>& search_ranges,
		       Tensor<int, 2, true>& search_frames,
		       cudaStream_t stream){
	int* one = (int*)malloc(sizeof(int));
	*one = 1;
	for (int i0 = 0; i0 < blocks.getSize(0); ++i0){
	  for (int i1 = 0; i1 < blocks.getSize(1); ++i1){
	    for (int i2 = 0; i2 < blocks.getSize(2); ++i2){
	      for (int i3 = 0; i3 < blocks.getSize(3); ++i3){
		for (int i4 = 0; i4 < blocks.getSize(4); ++i4){
		  cudaMemcpy(blocks[i0][i1][i2][i3][i4].data(),one,
			     sizeof(int),cudaMemcpyHostToDevice);
		}
	      }
	    }
	  }
	}
	free(one);
      }
    
      void test_case_1(Tensor<int, 5, true, int>& blocks,
		       Tensor<int, 3, true>& init_blocks,
		       Tensor<int, 5, true>& search_ranges,
		       Tensor<int, 2, true>& search_frames,
		       cudaStream_t stream){
	int iter = 0;
	create_search_space(search_ranges,blocks,init_blocks,
			    search_frames,iter,stream);

      }

    } // namespace test_mss

    //
    // Main Test Function 
    //

    void test_mesh_search_space(int test_case,
				Tensor<int, 5, true, int>& blocks,
				Tensor<int, 3, true>& init_blocks,
				Tensor<int, 5, true>& search_ranges,
				Tensor<int, 2, true>& search_frames,
				cudaStream_t stream){

      fprintf(stdout,"Testing: [mesh search space update]\n");
      if (test_case == 0){
	test_mss::test_case_0(blocks,init_blocks,search_ranges,search_frames,stream);
      }else if (test_case == 1){
	test_mss::test_case_1(blocks,init_blocks,search_ranges,search_frames,stream);
      }else{
	FAISS_THROW_FMT("[TestMeshSearchSpace.cu]: unimplemented test case %d",test_case);
      }

    }


  }
}