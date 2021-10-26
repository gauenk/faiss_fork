
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmCentroidUpdate.cuh>
#include <faiss/gpu/impl/TestCentroidUpdate.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>


namespace faiss {
  namespace gpu {

    namespace test_ceu{

      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<int, 5, true, int>& blocks,
		       Tensor<T, 5, true, int>& centroids,
		       Tensor<int, 4, true, int>& clusters,
		       Tensor<int, 1, true, int>& sizes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	T* one = (T*)malloc(sizeof(T));
	*one = 1;
	for (int i0 = 0; i0 < centroids.getSize(0); ++i0){
	  for (int i1 = 0; i1 < centroids.getSize(1); ++i1){
	    for (int i2 = 0; i2 < centroids.getSize(2); ++i2){
	      for (int i3 = 0; i3 < centroids.getSize(3); ++i3){
		for (int i4 = 0; i4 < centroids.getSize(4); ++i4){
		  cudaMemcpy(centroids[i0][i1][i2][i3][i4].data(),one,
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
		       Tensor<int, 4, true, int>& clusters,
		       Tensor<int, 1, true, int>& sizes,
		       int patchsize, float offset,
		       cudaStream_t stream){
	bool init_update = false;
	update_centroids(dists,burst,blocks,centroids,clusters,
			 sizes,init_update,stream);

      }
    
    } // namespace test_ceu

    //
    // Main Test Function 
    //

    template<typename T>
    void test_centroid_update(int test_case,
			      Tensor<T, 5, true, int>& dists,
			      Tensor<T, 4, true, int>& burst,
			      Tensor<int, 5, true, int>& blocks,
			      Tensor<T, 5, true, int>& centroids,
			      Tensor<int, 4, true, int>& clusters,
			      Tensor<int, 1, true, int>& sizes,
			      int patchsize, float offset,
			      cudaStream_t stream){

      fprintf(stdout,"Testing: [centroid update]\n");
      if (test_case == 0){
	test_ceu::test_case_0(dists, burst, blocks, centroids, clusters,
			      sizes, patchsize, offset, stream);
      }else if (test_case == 1){
	test_ceu::test_case_1(dists, burst, blocks, centroids, clusters,
			      sizes, patchsize, offset, stream);
      }else{
	FAISS_THROW_FMT("[TestCentroidUpdate.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_centroid_update(int test_case,
			      Tensor<float, 5, true, int>& dists,
			      Tensor<float, 4, true, int>& burst,
			      Tensor<int, 5, true, int>& blocks,
			      Tensor<float, 5, true, int>& centroids,
			      Tensor<int, 4, true, int>& clusters,
			      Tensor<int, 1, true, int>& sizes,
			      int patchsize, float offset,
			      cudaStream_t stream){
      test_centroid_update<float>(test_case,dists,burst,blocks,centroids,
				  clusters, sizes, patchsize, offset, stream);
    }

    void test_centroid_update(int test_case,
			      Tensor<half, 5, true, int>& dists,
			      Tensor<half, 4, true, int>& burst,
			      Tensor<int, 5, true, int>& blocks,
			      Tensor<half, 5, true, int>& centroids,
			      Tensor<int, 4, true, int>& clusters,
			      Tensor<int, 1, true, int>& sizes,
			      int patchsize, float offset,
			      cudaStream_t stream){
      test_centroid_update<half>(test_case,dists,burst,blocks,centroids,
				 clusters, sizes, patchsize, offset, stream);

    }

  }
}