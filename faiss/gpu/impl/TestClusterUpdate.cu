
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmClusterUpdate.cuh>
#include <faiss/gpu/impl/TestClusterUpdate.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <algorithm>


namespace faiss {
  namespace gpu {

    namespace test_clu{
      //
      // Test cases
      //

      template<typename T>
      void test_case_0(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<uint8_t, 4, true, int>& sizes,
		       float offset, cudaStream_t stream){
	fprintf(stdout,"[clusters.] test_case_0\n");
	int* one = (int*)malloc(sizeof(int));
	*one = 1;
	for (int i0 = 0; i0 < clusters.getSize(0); ++i0){
	  for (int i1 = 0; i1 < clusters.getSize(1); ++i1){
	    for (int i2 = 0; i2 < clusters.getSize(2); ++i2){
	      for (int i3 = 0; i3 < clusters.getSize(3); ++i3){
		cudaMemcpy(clusters[i0][i1][i2][i3].data(),one,
			   sizeof(int),cudaMemcpyHostToDevice);
	      }
	    }
	  }
	}
	free(one);
      }

      template<typename T>
      void test_case_1(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<uint8_t, 4, true, int>& sizes,
		       float offset, cudaStream_t stream){
	bool init_update = true;
	update_clusters(dists, burst, clusters, sizes, init_update, stream);
      }

      template<typename T>
      void test_case_2(Tensor<T, 5, true, int>& dists,
		       Tensor<T, 4, true, int>& burst,
		       Tensor<uint8_t, 4, true, int>& clusters,
		       Tensor<uint8_t, 4, true, int>& sizes,
		       float offset, cudaStream_t stream){
	bool init_update = false;
	update_clusters(dists, burst, clusters, sizes, init_update, stream);
      }

    
    } // namespace test_clu

    //
    // Main Test Function 
    //

    template<typename T>
    void test_cluster_update(int test_case,
			     Tensor<T, 5, true, int>& dists,
			     Tensor<T, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<uint8_t, 4, true, int>& sizes,
			     float offset, cudaStream_t stream){

      fprintf(stdout,"Testing: [centroid update]\n");
      if (test_case == 0){
	test_clu::test_case_0(dists,burst,clusters,sizes,offset,stream);
      }else if (test_case == 1){
	test_clu::test_case_1(dists,burst,clusters,sizes,offset,stream);
      }else if (test_case == 2){
	test_clu::test_case_2(dists,burst,clusters,sizes,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestClusterUpdate.cu]: unimplemented test case %d",test_case);
      }

    }

    //
    // Template Init
    // 

    void test_cluster_update(int test_case,
			     Tensor<float, 5, true, int>& dists,
			     Tensor<float, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<uint8_t, 4, true, int>& sizes,
			     float offset, cudaStream_t stream){
      test_cluster_update<float>(test_case, dists, burst,
				 clusters, sizes, offset, stream);
    }

    void test_cluster_update(int test_case,
			     Tensor<half, 5, true, int>& dists,
			     Tensor<half, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     Tensor<uint8_t, 4, true, int>& sizes,
			     float offset, cudaStream_t stream){
      test_cluster_update<half>(test_case, dists, burst,
				clusters, sizes, offset, stream);

    }

  }
}