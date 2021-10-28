

#pragma once

#include <faiss/gpu/TestResources.h>
#include <faiss/gpu/impl/GeneralDistance.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
  namespace gpu {

    class GpuResources;

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
			Tensor<float, 1, true, int>& modes,
			Tensor<float, 4, true, int>& ave);
    
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
			Tensor<float, 1, true, int>& modes,
			Tensor<half, 4, true, int>& ave);



    template <typename T>
    void kmBurstTestOnDevice(GpuResources* resources,
			     int device,
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
			     Tensor<float, 1, true, int>& modes,
			     Tensor<T, 4, true, int>& ave){
      // setup device
      DeviceScope ds(device);
      // We are guaranteed that all data arguments are resident on our preferred
      // `device` here

      runKmBurstTest(resources,stream,burst,search_ranges,search_frames,
		     init_blocks,test_type,test_case,kmeansK,nsiters,k,t,h,w,
		     c,ps,nbsearch,nfsearch,std,outDistances,outIndices,
		     dists,self_dists,blocks,centroids,clusters,
		     cluster_sizes,modes,ave);
    }
    
  }
}

