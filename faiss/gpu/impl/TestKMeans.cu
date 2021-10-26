
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/TestKMeans.cuh>
#include <faiss/gpu/impl/KMeans.cuh>
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

    namespace test_km{
      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<int, 4, true, int>& clusters,
		       Tensor<int, 1, true, int>& cluster_sizes,
		       int patchsize, float offset, int kmeansK,
		       cudaStream_t stream){
	thrust::fill(thrust::cuda::par.on(stream), centroids.data(),
		     centroids.end(),1);
	thrust::fill(thrust::cuda::par.on(stream), clusters.data(),
		     clusters.end(),1);
	thrust::fill(thrust::cuda::par.on(stream), dists.data(),
		     dists.end(),1);
      }
    
      template<typename T>
      void test_case_1(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<int, 4, true, int>& clusters,
		       Tensor<int, 1, true, int>& cluster_sizes,
		       int patchsize, float offset, int kmeansK,
		       cudaStream_t stream){
	kmeans_clustering(dists,burst,blocks,centroids,
			  clusters,cluster_sizes,patchsize,
			  kmeansK,offset,stream);
      }

    } // namespace test_means

    //
    // Main Test Function 
    //

    template<typename T>
    void test_kmeans(int test_case,
		     Tensor<T, 5, true, int>& dists,
		     Tensor<T, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<T, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& cluster_sizes,
		     int patchsize, float offset, int kmeansK,
		     cudaStream_t stream){

      fprintf(stdout,"Testing: [kmeans]\n");
      if (test_case == 0){
	test_km::test_case_0<T>(dists,burst,blocks,centroids,clusters,
				cluster_sizes,patchsize,offset,kmeansK,stream);
      }else if (test_case == 1){
	test_km::test_case_1<T>(dists,burst,blocks,centroids,clusters,
				cluster_sizes,patchsize,offset,kmeansK,stream);
      }else{
	FAISS_THROW_FMT("[TestKMeans.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_kmeans(int test_case,
		     Tensor<float, 5, true, int>& dists,
		     Tensor<float, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<float, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& cluster_sizes,
		     int patchsize, float offset, int kmeansK,
		     cudaStream_t stream){
      test_kmeans<float>(test_case,dists,burst,blocks,centroids,clusters,
			 cluster_sizes, patchsize, offset, kmeansK, stream);
    }

    void test_kmeans(int test_case,
		     Tensor<half, 5, true, int>& dists,
		     Tensor<half, 4, true, int>& burst,
		     Tensor<int, 5, true, int>& blocks,
		     Tensor<half, 5, true, int>& centroids,
		     Tensor<int, 4, true, int>& clusters,
		     Tensor<int, 1, true, int>& cluster_sizes,
		     int patchsize, float offset, int kmeansK,
		     cudaStream_t stream){
      test_kmeans<half>(test_case,dists,burst,blocks,centroids,clusters,
			cluster_sizes, patchsize, offset, kmeansK, stream);

    }

  }
}