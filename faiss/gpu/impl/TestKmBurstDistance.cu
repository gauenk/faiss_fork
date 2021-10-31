
#include <faiss/gpu/TestResources.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/BroadcastSumBurst.cuh>
#include <faiss/gpu/impl/BurstPatchDistance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/MeshSearchSpace.cuh>
#include <faiss/gpu/impl/KmUtils.cuh>
#include <faiss/gpu/impl/KMeans.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/BurstNnfL2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/impl/KmBurstTopK.cuh>
#include <faiss/gpu/utils/BurstBlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>
#include <faiss/gpu/utils/BlockIndices2Labels.cuh>
#include <faiss/gpu/impl/TestPairwiseDistances.cuh>
#include <faiss/gpu/impl/TestClusterUpdate.cuh>
#include <faiss/gpu/impl/TestCentroidUpdate.cuh>
#include <faiss/gpu/impl/TestMeshSearchSpace.cuh>
#include <faiss/gpu/impl/TestKMeans.cuh>
#include <faiss/gpu/impl/TestComputeMode.cuh>
#include <faiss/gpu/impl/TestKmBurstAve.cuh>
#include <faiss/gpu/impl/TestKmBurstL2Norm.cuh>
#include <faiss/gpu/impl/TestKmBurstTopK.cuh>

#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>


namespace faiss {
  namespace gpu {

    template<typename T>
    void runKmBurstTest(GpuResources* resources,
			cudaStream_t stream,
			Tensor<T, 4, true>& burst,
			Tensor<int, 5, true>& search_ranges,
			Tensor<int, 2, true>& search_frames,
			Tensor<int, 3, true>& init_blocks,
			KM_TEST_TYPE test_type,
			int test_case,
			int kmeansK,
			int nsiters,
			int k,
			int t,
			int h,
			int w,
			int c,
			int ps,
			int nbsearch,
			int nfsearch,
			float std,
			Tensor<float, 3, true>& outDistances,
			Tensor<int, 5, true>& outIndices,
			Tensor<T, 5, true, int>& dists,
			Tensor<T, 5, true, int>& self_dists,
			Tensor<int, 5, true, int>& blocks,
			Tensor<T, 5, true, int>& centroids,
			Tensor<uint8_t, 4, true, int>& clusters,
			Tensor<uint8_t, 4, true, int>& cluster_sizes,
			Tensor<T, 4, true, int>& modes,
			Tensor<T, 3, true, int>& modes3d,
			Tensor<T, 4, true, int>& ave,
			Tensor<float, 3, true, int>& vals){

      // Select the test here.
      fprintf(stdout,"\nrunKmBurstTest :D\n");
      float offset = 0;
      if (test_type == PairwiseDistanceCase){
	test_pairwise_distances(test_case,dists,self_dists,burst,
				blocks,centroids,ps,offset,stream);
      }else if(test_type == ClusterUpdate){
	test_cluster_update(test_case,dists,burst,clusters,
			    cluster_sizes,offset,stream);
      }else if(test_type == CentroidUpdate){
	test_centroid_update(test_case,dists,burst,blocks,centroids,
			     clusters,cluster_sizes,ps,offset,stream);
      }else if(test_type == Ranges2Blocks){
	test_mesh_search_space(test_case,blocks,init_blocks,
			       search_ranges,search_frames,stream);
      }else if(test_type == KMeansCase){
	test_kmeans(test_case,dists,burst,blocks,centroids,
		    clusters,cluster_sizes,ps,offset,kmeansK,stream);
      }else if(test_type == ComputeModeCase){
	test_compute_mode(test_case,dists,burst,blocks,centroids,
			  clusters,cluster_sizes,modes,ps,offset,std,stream);
      }else if(test_type == KmBurstAveCase){
	test_kmburst_ave(test_case,dists,burst,blocks,centroids,
			 clusters,ave,modes,ps,offset,stream);
      }else if(test_type == KmBurstL2NormCase){
	test_kmburst_l2norm(test_case,outDistances,burst,blocks,
			    centroids,clusters,ave,modes,ps,offset,stream);
      }else if(test_type == KmBurstTopKCase){
	test_kmburst_topK(test_case,outDistances,outIndices,burst,blocks,
			  centroids,clusters,ave,modes,modes3d,vals,ps,offset,stream);
      }else{
	FAISS_THROW_FMT("[TestKmBurstDistance]: unimplemented test %d", test_type);
      }
      
    }

    void runKmBurstTest(GpuResources* resources,
			cudaStream_t stream,
			Tensor<float, 4, true>& burst,
			Tensor<int, 5, true>& search_ranges,
			Tensor<int, 2, true>& search_frames,
			Tensor<int, 3, true>& init_blocks,
			KM_TEST_TYPE test_type,
			int test_case,
			int kmeansK,
			int nsiters,
			int k,
			int t,
			int h,
			int w,
			int c,
			int ps,
			int nbsearch,
			int nfsearch,
			float std,
			Tensor<float, 3, true>& outDistances,
			Tensor<int, 5, true>& outIndices,
			Tensor<float, 5, true, int>& dists,
			Tensor<float, 5, true, int>& self_dists,
			Tensor<int, 5, true, int>& blocks,
			Tensor<float, 5, true, int>& centroids,
			Tensor<uint8_t, 4, true, int>& clusters,
			Tensor<uint8_t, 4, true, int>& cluster_sizes,
			Tensor<float, 4, true, int>& modes,
			Tensor<float, 3, true, int>& modes3d,
			Tensor<float, 4, true, int>& ave,
			Tensor<float, 3, true, int>& vals){
      runKmBurstTest<float>(resources,stream,burst,search_ranges,
			    search_frames,init_blocks,test_type,
			    test_case,kmeansK,nsiters,k,t,h,w,c,ps,
			    nbsearch,nfsearch,std,
			    outDistances,outIndices,dists,self_dists,
			    blocks,centroids,clusters,
			    cluster_sizes,modes,modes3d,ave,vals);
    }
    
    void runKmBurstTest(GpuResources* resources,
			cudaStream_t stream,
			Tensor<half, 4, true>& burst,
			Tensor<int, 5, true>& search_ranges,
			Tensor<int, 2, true>& search_frames,
			Tensor<int, 3, true>& init_blocks,
			KM_TEST_TYPE test_type,
			int test_case,
			int kmeansK,
			int nsiters,
			int k,
			int t,
			int h,
			int w,
			int c,
			int ps,
			int nbsearch,
			int nfsearch,
			float std,
			Tensor<float, 3, true>& outDistances,
			Tensor<int, 5, true>& outIndices,
			Tensor<half, 5, true, int>& dists,
			Tensor<half, 5, true, int>& self_dists,
			Tensor<int, 5, true, int>& blocks,
			Tensor<half, 5, true, int>& centroids,
			Tensor<uint8_t, 4, true, int>& clusters,
			Tensor<uint8_t, 4, true, int>& cluster_sizes,
			Tensor<half, 4, true, int>& modes,
			Tensor<half, 3, true, int>& modes3d,
			Tensor<half, 4, true, int>& ave,
			Tensor<float, 3, true, int>& vals){
      runKmBurstTest<half>(resources,stream,burst,search_ranges,
			   search_frames,init_blocks,test_type,
			   test_case,kmeansK,nsiters,k,t,h,w,c,ps,
			   nbsearch,nfsearch,std,
			   outDistances,outIndices,dists,self_dists,
			   blocks,centroids,clusters,
			   cluster_sizes,modes,modes3d,ave,vals);
    }

  }
}