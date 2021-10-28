
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/PairwiseDistances.cuh>
#include <faiss/gpu/impl/SelfPairwiseDistances.cuh>
#include <faiss/gpu/impl/TestPairwiseDistances.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>


namespace faiss {
  namespace gpu {

    namespace test_pwd { // namespace for test cases
      
      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       int patchsize, float offset,
		       cudaStream_t stream){
	T* one = (T*)malloc(sizeof(T));
	*one = 1;
	for (int i0 = 0; i0 < dists.getSize(0); ++i0){
	  for (int i1 = 0; i1 < dists.getSize(1); ++i1){
	    for (int i2 = 0; i2 < dists.getSize(2); ++i2){
	      for (int i3 = 0; i3 < dists.getSize(3); ++i3){
		for (int i4 = 0; i4 < dists.getSize(4); ++i4){
		  cudaMemcpy(dists[i0][i1][i2][i3][i4].data(),one,
			     sizeof(T),cudaMemcpyHostToDevice);
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
		       int patchsize, float offset,
		       cudaStream_t stream){
	self_pairwise_distances(dists,burst,blocks,patchsize,offset,stream);
      }

      template<typename T>
      void test_case_2(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       int patchsize, float offset,
		       cudaStream_t stream){
	pairwise_distances(dists,burst,blocks,centroids,
			   patchsize,offset,stream);
      }

    } // namespace test_pwd
    

    //
    // Main Test Function 
    //

    template<typename T>
    void test_pairwise_distances(int test_case,
				 Tensor<T, 5, true, int>& dists,
				 Tensor<T, 5, true, int>& self_dists,
				 Tensor<T, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 Tensor<T, 5, true, int>& centroids,
				 int patchsize, float offset,
				 cudaStream_t stream){

      fprintf(stdout,"Testing: [pairwise dists.]\n");
      if (test_case == 0){
	test_pwd::test_case_0<T>(dists,burst,blocks,centroids,
				 patchsize,offset,stream);
      }else if (test_case == 1){
	test_pwd::test_case_1<T>(self_dists,burst,blocks,centroids,
				 patchsize,offset,stream);
      }else if (test_case == 2){
	test_pwd::test_case_2<T>(dists,burst,blocks,centroids,
				 patchsize,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestPairwiseDistances.cu]: unimplemented test case %d",
			test_case);
      }
    }


    //
    // Template Inits
    //
    
    void test_pairwise_distances(int test_case,
				 Tensor<float, 5, true, int>& dists,
				 Tensor<float, 5, true, int>& self_dists,
				 Tensor<float, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 Tensor<float, 5, true, int>& centroids,
				 int patchsize, float offset,
				 cudaStream_t stream){
      test_pairwise_distances<float>(test_case,dists,self_dists,
				     burst,blocks,centroids,
				     patchsize, offset, stream);
    }

    void test_pairwise_distances(int test_case,
				 Tensor<half, 5, true, int>& dists,
				 Tensor<half, 5, true, int>& self_dists,
				 Tensor<half, 4, true, int>& burst,
				 Tensor<int, 5, true, int>& blocks,
				 Tensor<half, 5, true, int>& centroids,
				 int patchsize, float offset,
				 cudaStream_t stream){
      test_pairwise_distances<half>(test_case,dists,self_dists,
				    burst,blocks,centroids,
				    patchsize, offset, stream);
    }

  }
}