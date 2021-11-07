
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmBurstL2Norm.cuh>
#include <faiss/gpu/impl/TestKmBurstL2Norm.cuh>
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

    namespace test_kmb_l2norm{

      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<float, 3, true, int>& outDists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<T, 4, true, int>& ave,
		       Tensor<T, 4, true, int>& modes,
		       Tensor<float, 3, true, int>& vals,
		       int patchsize, float offset,
		       cudaStream_t stream){
	thrust::fill(thrust::cuda::par.on(stream),
		     vals.data(), vals.end(),1);
      }
    
      template<typename T>
      void test_case_1(Tensor<float, 3, true, int>& outDists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<T, 4, true, int>& ave,
		       Tensor<T, 4, true, int>& modes,
		       Tensor<float, 3, true, int>& vals,
		       int patchsize, float offset,
		       cudaStream_t stream){
	runKmBurstL2Norm(centroids,ave,blocks,
			 vals,patchsize,
			 1.,true,stream);
      }

    } // namespace test_kmb_l2norm

    //
    // Main Test Function 
    //

    template<typename T>
    void test_kmburst_l2norm(int test_case,
			     Tensor<float, 3, true, int>& outDists,
			     Tensor<T, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<T, 5, true, int>& centroids,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<T, 4, true, int>& ave,
			     Tensor<T, 4, true, int>& modes,
			     Tensor<float, 3, true, int>& vals,
			     int patchsize, float offset,
			     cudaStream_t stream){

      fprintf(stdout,"Testing: [km burst l2norm]\n");
      if (test_case == 0){
	test_kmb_l2norm::test_case_0<T>(outDists,burst,blocks,centroids,clusters,
					ave,modes,vals,patchsize,offset,stream);
      }else if (test_case == 1){
	test_kmb_l2norm::test_case_1<T>(outDists,burst,blocks,centroids,clusters,
					ave,modes,vals,patchsize,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestKmBurstL2Norm.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_kmburst_l2norm(int test_case,
			     Tensor<float, 3, true, int>& outDists,
			     Tensor<float, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<float, 5, true, int>& centroids,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<float, 4, true, int>& ave,
			     Tensor<float, 4, true, int>& modes,
			     Tensor<float, 3, true, int>& vals,
			     int patchsize, float offset,
			     cudaStream_t stream){
      test_kmburst_l2norm<float>(test_case, outDists, burst,
				 blocks, centroids, clusters,
				 ave, modes, vals,
				 patchsize, offset, stream);
    }

    void test_kmburst_l2norm(int test_case,
			     Tensor<float, 3, true, int>& outDists,
			     Tensor<half, 4, true, int>& burst,
			     Tensor<int, 5, true, int>& blocks,
			     Tensor<half, 5, true, int>& centroids,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<half, 4, true, int>& ave,
			     Tensor<half, 4, true, int>& modes,
			     Tensor<float, 3, true, int>& vals,
			     int patchsize, float offset,
			     cudaStream_t stream){
      test_kmburst_l2norm<half>(test_case, outDists, burst,
				blocks,centroids, clusters,
				ave, modes, vals,
				patchsize, offset, stream);

    }

  }
}