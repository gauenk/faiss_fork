
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmUtils.cuh>
#include <faiss/gpu/impl/TestComputeMode.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>
#include <thrust/fill.h>


namespace faiss {
  namespace gpu {

    namespace test_cmode{
      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<float, 1, true, int>& modes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	thrust::fill(thrust::cuda::par.on(stream),
		     modes.data(), modes.end(),1);
      }
    
      template<typename T>
      void test_case_1(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<float, 1, true, int>& modes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	fprintf(stdout,"test_case_1,\n");
      }

    } // namespace test_cmode

    //
    // Main Test Function 
    //

    template<typename T>
    void test_compute_mode(int test_case,
			   Tensor<T, 5, true, int>& dists,
			   Tensor<T, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<T, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<float, 1, true, int>& modes,
			   int patchsize, float offset,
			   cudaStream_t stream){

      fprintf(stdout,"Testing: [compute mode]\n");
      if (test_case == 0){
	test_cmode::test_case_0<T>(dists,burst,blocks,centroids,clusters,
				   modes,patchsize,offset,stream);
      }else if (test_case == 1){
	test_cmode::test_case_1<T>(dists,burst,blocks,centroids,clusters,
				   modes,patchsize,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestComputeMode.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_compute_mode(int test_case,
			   Tensor<float, 5, true, int>& dists,
			   Tensor<float, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<float, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<float, 1, true, int>& modes,
			   int patchsize, float offset,
			   cudaStream_t stream){
      test_compute_mode<float>(test_case,dists,burst,blocks,centroids,
			       clusters, modes, patchsize, offset, stream);
    }

    void test_compute_mode(int test_case,
			   Tensor<half, 5, true, int>& dists,
			   Tensor<half, 4, true, int>& burst,
			   Tensor<int, 5, true, int>& blocks,
			   Tensor<half, 5, true, int>& centroids,
			   Tensor<uint8_t, 4, true, int>& clusters,
			   Tensor<float, 1, true, int>& modes,
			   int patchsize, float offset,
			   cudaStream_t stream){
      test_compute_mode<half>(test_case,dists,burst,blocks,centroids,
			      clusters, modes, patchsize, offset, stream);

    }

  }
}