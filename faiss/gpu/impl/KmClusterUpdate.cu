

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmClusterUpdate.cuh>
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

    /************************************

          Update Cluster Sizes

    *************************************/
    __global__ void update_sizes_kernel(Tensor<uint8_t, 4, true, int> clusters,
					Tensor<uint8_t, 4, true, int> sizes){

      // height,width
      int h = blockIdx.x;
      int w = blockIdx.y;

      // get dims to comp indices from thread
      int nframes = clusters.getSize(0);
      int nclusters = sizes.getSize(0);
      int nblocks = sizes.getSize(1);
      int dim = nclusters*nblocks;

      // helpers
      int b,cid,num;

      // set clusters
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){
	cid = tIdx % nclusters;
	b = (tIdx / nclusters) % nblocks;
	for (int t = 0; t < nframes; ++t){
	  num = clusters[t][b][h][w] == cid ? 1 : 0;
	  sizes[cid][b][h][w] += num;
	}
      }
    }

    void update_sizes(Tensor<uint8_t, 4, true, int>& clusters,
		      Tensor<uint8_t, 4, true, int>& sizes,
		      cudaStream_t stream){
      // shapes
      int nframes = clusters.getSize(0);
      int bBatch = clusters.getSize(1);
      int hBatch = clusters.getSize(2);
      int wBatch = clusters.getSize(3);
      int nclusters = sizes.getSize(0);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nclusters*bBatch;
      int numThreads = std::min(dim, maxThreads);

      // launch
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch kernel 
      update_sizes_kernel<<<grid,block,0,stream>>>(clusters,sizes);
    }

    /******************************

            Initial Clustering

     ******************************/
    template <bool NormLoop>
    __global__ void init_clusters_kernel(Tensor<uint8_t, 4, true, int> clusters,
					 int nclusters){
      // get shapes
      int nframes = clusters.getSize(0);
      int nblocks = clusters.getSize(1);

      // set blocks
      int h = blockIdx.x;
      int w = blockIdx.y;

      // setup helpers
      int t,b,tDiv;
      uint8_t cid;
      int dim = nframes*nblocks;

      // set to init vals
      if (NormLoop){
	for(int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){
	  t = tIdx % nframes;
	  tDiv = tIdx / nframes;
	  b = tDiv % nblocks;
	  cid = t % nclusters;
	  clusters[t][b][h][w] = cid;
	}
      }else{
	t = threadIdx.x % nframes;
	tDiv = threadIdx.x / nframes;
	b = tDiv % nblocks;
	cid = t % nclusters;
	clusters[t][b][h][w] = cid;
      }
    }

    void run_init_clusters(Tensor<uint8_t, 4, true, int>& clusters,
			   int nclusters, cudaStream_t stream){

      // shapes
      int nframes = clusters.getSize(0);
      int bBatch = clusters.getSize(1);
      int hBatch = clusters.getSize(2);
      int wBatch = clusters.getSize(3);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nframes*bBatch;
      bool NormLoop = dim > maxThreads;
      int numThreads = std::min(dim, maxThreads);

      // launch
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch kernel
      if (NormLoop){
	init_clusters_kernel<true>
	  <<<grid,block,0,stream>>>(clusters,nclusters);
      }else{
	init_clusters_kernel<false>
	  <<<grid,block,0,stream>>>(clusters,nclusters);
      }
    }

    /************************************

        Cluster from Pairwise Distance   

    *************************************/
    template <typename T>
    __global__ void update_clusters_kernel(Tensor<T, 5, true, int> dists,
					   Tensor<uint8_t, 4, true, int> clusters){
      // helpers
      T d,d_min;
      int ci,t,b;
      
      // sizes
      int nframes = dists.getSize(0);
      int nclusters = dists.getSize(1);
      int nblocks = dists.getSize(2);
      int dim = nframes*nblocks;
      
      // height and width
      int h = blockIdx.x;
      int w = blockIdx.y;

      // loop over threads as necessary
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){

	// fix (frames) and (blocks)
       	t = tIdx % nframes;
	b = (tIdx / nframes) % nblocks;
	
	// find nearest cluster
	ci = 0;
	d_min = 10000;
	for (int c_iter = 0; c_iter < nclusters; ++c_iter){
	  d = dists[t][c_iter][b][h][w];
	  if (d < d_min){
	    ci = c_iter;
	    d_min = d;
	  }
	}
	clusters[t][b][h][w] = ci;

      }
    }


    template <typename T>
    void run_update_clusters(Tensor<T, 5, true, int>& dists,
			     Tensor<T, 4, true, int>& burst,
			     Tensor<uint8_t, 4, true, int>& clusters,
			     cudaStream_t stream){

      // shapes
      int nframes = dists.getSize(0);
      int nclusters = dists.getSize(1);
      int nblocks = dists.getSize(1);
      int bBatch = dists.getSize(1);
      int hBatch = dists.getSize(3);
      int wBatch = dists.getSize(4);
      // int nftrs = burst.getSize(0);
      // int height = burst.getSize(2);
      // int width = burst.getSize(3);
      // int nframes = clusters.getSize(0);
      // int bBatch = clusters.getSize(1);
      // int hBatch = clusters.getSize(2);
      // int wBatch = clusters.getSize(3);

      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nframes*nblocks;
      int numThreads = std::min(dim, maxThreads);

      // launch params
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch kernel
      update_clusters_kernel<T><<<grid,block,0,stream>>>(dists,clusters);

    }

    /************************************

          Interface Function Call

    *************************************/

    template <typename T>
    void update_clusters(Tensor<T, 5, true, int>& dists,
			 Tensor<T, 4, true, int>& burst,
			 Tensor<uint8_t, 4, true, int>& clusters,
			 Tensor<uint8_t, 4, true, int>& sizes,
			 bool init_update, cudaStream_t stream){
      // reset sizes
      thrust::fill(thrust::cuda::par.on(stream),
		   sizes.data(),sizes.end(),0);

      // update clusters
      if (init_update){
	int nclusters = sizes.getSize(0);
	run_init_clusters(clusters,nclusters,stream);
      }else{
	run_update_clusters(dists,burst,clusters,stream);
      }
      CUDA_TEST_ERROR();

      // update sizes 
      update_sizes(clusters,sizes,stream);

    }


    //
    // Template Inits
    //
    
    void update_clusters(Tensor<float, 5, true, int>& dists,
			 Tensor<float, 4, true, int>& burst,
			 Tensor<uint8_t, 4, true, int>& clusters,
			 Tensor<uint8_t, 4, true, int>& sizes,
			 bool init_update, cudaStream_t stream){
      update_clusters<float>(dists,burst,clusters,sizes,init_update,stream);
    }

    void update_clusters(Tensor<half, 5, true, int>& dists,
			 Tensor<half, 4, true, int>& burst,
			 Tensor<uint8_t, 4, true, int>& clusters,
			 Tensor<uint8_t, 4, true, int>& sizes,
			 bool init_update, cudaStream_t stream){
      update_clusters<half>(dists,burst,clusters,sizes,init_update,stream);
    }


  }
}
