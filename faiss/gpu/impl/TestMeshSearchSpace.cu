
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

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<int, 4, true, int>& clusters,
		       int patchsize, float offset,
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
    
      template<typename T>
      void test_case_1(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<int, 4, true, int>& clusters,
		       int patchsize, float offset,
		       cudaStream_t stream){
	fprintf(stdout,"test_case_1,\n");
      }

    } // namespace test_mss

    //
    // Main Test Function 
    //

    template<typename T>
    void test_mesh_search_space(int test_case,
				Tensor<T, 5, true, int>& dists,
				Tensor<T, 4, true, int>& burst,
				Tensor<int, 5, true, int>& blocks,
				Tensor<T, 5, true, int>& centroids,
				Tensor<int, 4, true, int>& clusters,
				int patchsize, float offset,
				cudaStream_t stream){

      fprintf(stdout,"Testing: [mesh search space update]\n");
      if (test_case == 0){
	test_mss::test_case_0<T>(dists,burst,blocks,centroids,
				 clusters,patchsize,offset,stream);
      }else if (test_case == 1){
	test_mss::test_case_1<T>(dists,burst,blocks,centroids,
				 clusters,patchsize,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestMeshSearchSpace.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_mesh_search_space(int test_case,
				Tensor<float, 5, true, int>& dists,
				Tensor<float, 4, true, int>& burst,
				Tensor<int, 5, true, int>& blocks,
				Tensor<float, 5, true, int>& centroids,
				Tensor<int, 4, true, int>& clusters,
				int patchsize, float offset,
				cudaStream_t stream){
      test_mesh_search_space<float>(test_case,dists,burst,blocks,centroids,clusters,
				    patchsize, offset, stream);
    }

    void test_mesh_search_space(int test_case,
				Tensor<half, 5, true, int>& dists,
				Tensor<half, 4, true, int>& burst,
				Tensor<int, 5, true, int>& blocks,
				Tensor<half, 5, true, int>& centroids,
				Tensor<int, 4, true, int>& clusters,
				int patchsize, float offset,
				cudaStream_t stream){
      test_mesh_search_space<half>(test_case,dists,burst,blocks,centroids,clusters,
				   patchsize, offset, stream);

    }

  }
}