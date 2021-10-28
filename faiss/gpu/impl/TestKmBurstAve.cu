
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>

// TODO: add KmBurstTopAve.cuh 

#include <faiss/gpu/impl/TestKmBurstAve.cuh>
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

    namespace test_kmbave{
      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<T, 4, true, int>& ave,
		       Tensor<float, 1, true, int>& modes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	thrust::fill(thrust::cuda::par.on(stream),
		     ave.data(), ave.end(),1);
      }
    
      template<typename T>
      void test_case_1(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<T, 4, true, int>& ave,
		       Tensor<float, 1, true, int>& modes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	fprintf(stdout,"test_case_1.\n");
      }

    } // namespace test_kmbave

    //
    // Main Test Function 
    //

    template<typename T>
    void test_kmburst_ave(int test_case,
			  Tensor<T, 5, true, int>& dists,
			  Tensor<T, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<T, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<T, 4, true, int>& ave,
			  Tensor<float, 1, true, int>& modes,
			  int patchsize, float offset,
			  cudaStream_t stream){

      fprintf(stdout,"Testing: [km burst ave]\n");
      if (test_case == 0){
	test_kmbave::test_case_0(dists,burst,blocks,centroids,clusters,
				 ave,modes,patchsize,offset,stream);
      }else if (test_case == 1){
	test_kmbave::test_case_1(dists,burst,blocks,centroids,clusters,
				 ave,modes,patchsize,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestKmBurstAve.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_kmburst_ave(int test_case,
			  Tensor<float, 5, true, int>& dists,
			  Tensor<float, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<float, 4, true, int>& ave,
			  Tensor<float, 1, true, int>& modes,
			  int patchsize, float offset,
			  cudaStream_t stream){
      test_kmburst_ave<float>(test_case,dists,burst,blocks,centroids, clusters,
			      ave, modes, patchsize, offset, stream);
    }

    void test_kmburst_ave(int test_case,
			  Tensor<half, 5, true, int>& dists,
			  Tensor<half, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<half, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<half, 4, true, int>& ave,
			  Tensor<float, 1, true, int>& modes,
			  int patchsize, float offset,
			  cudaStream_t stream){
      test_kmburst_ave<half>(test_case,dists,burst,blocks,centroids, clusters,
			     ave, modes, patchsize, offset, stream);

    }

  }
}