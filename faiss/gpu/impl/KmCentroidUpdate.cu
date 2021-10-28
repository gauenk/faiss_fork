

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


namespace faiss {
  namespace gpu {

    __forceinline__ __device__
    int hw_boundary(int hw, int max){
      hw = (hw) < 0 ? -(hw) : (hw);
      hw = (hw) > (max) ? (2*(max) - (hw)) : (hw);
      return hw;
    }

    template <typename T>
    __global__ void init_centroids_kernel(Tensor<T, 5, true, int> dists,
					    Tensor<T, 4, true, int> burst,
					    Tensor<int, 5, true, int> blocks,
					    Tensor<T, 5, true, int> centroids,
					    Tensor<uint8_t, 4, true, int> clusters,
					    Tensor<uint8_t, 4, true, int> sizes){
      // compute pairwise distances across time.
      
    }



    template <typename T>
    __global__ void update_centroids_kernel(Tensor<T, 5, true, int> dists,
					    Tensor<T, 4, true, int> burst,
					    Tensor<int, 5, true, int> indices,
					    Tensor<T, 5, true, int> centroids,
					    Tensor<uint8_t, 4, true, int> clusters,
					    Tensor<uint8_t, 4, true, int> sizes){
      // helpers
      T val;
      int c,t,b,f,bH,bW,size;
      
      // sizes
      int nftrs = burst.getSize(0);
      int nframes = dists.getSize(0);
      int nclusters = dists.getSize(1);
      int nblocks = dists.getSize(2);
      int dim = nftrs*nclusters*nblocks;
      int height = burst.getSize(2);
      int width = burst.getSize(3);
      
      // height and width
      int h = blockIdx.x;
      int w = blockIdx.y;

      // loop over threads as necessary
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){

	// fix (cluster) and (blocks) and (feature)
       	c = tIdx % nclusters;
	b = (tIdx / nclusters) % nblocks;
	f = ((tIdx / nclusters)/nblocks) % nftrs;
	centroids[f][c][b][h][w] = 0;
	
	
	// sum over clusters
	for (int t_iter = 0; t_iter < nframes; ++t_iter){

	  // read indices of images
	  bH = indices[0][t_iter][b][h][w];
	  bW = indices[1][t_iter][b][h][w];
	  bH = hw_boundary(bH,height-1);
	  bW = hw_boundary(bW,width-1);

	  val = burst[f][t_iter][bH][bW];

	  if (clusters[t_iter][b][h][w] == c){
	    centroids[f][c][b][h][w] += val;
	  }

	}
	
	// divide by size
	size = sizes[c][b][h][w];
	val = centroids[f][c][b][h][w];
	if (size != 0){
	  centroids[f][c][b][h][w] = val / ((T)size);
	} else{
	  centroids[f][c][b][h][w] = 0;
	}

      }
      
    }


    /************************************

         Primary Centroid Update

    *************************************/


    template <typename T>
    void run_update_centroids(Tensor<T, 5, true, int>& dists,
			      Tensor<T, 4, true, int>& burst,
			      Tensor<int, 5, true, int>& blocks,
			      Tensor<T, 5, true, int>& centroids,
			      Tensor<uint8_t, 4, true, int>& clusters,
			      Tensor<uint8_t, 4, true, int>& sizes,
			      cudaStream_t stream){

      // shapes
      int nftrs = burst.getSize(0);
      int nframes = dists.getSize(0);
      int nclusters = dists.getSize(1);
      int nblocks = dists.getSize(1);
      int bBatch = dists.getSize(1);
      int hBatch = dists.getSize(3);
      int wBatch = dists.getSize(4);
      
      // threads
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = nftrs*nclusters*nblocks;
      int numThreads = std::min(dim, maxThreads);

      // launch params
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch
      update_centroids_kernel<T><<<grid,block,0,stream>>>(dists,burst,blocks,
							  centroids,clusters,sizes);
    }

    /************************************

          Interface Function Call

    *************************************/

    template <typename T>
    void update_centroids(Tensor<T, 5, true, int>& dists,
			  Tensor<T, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<T, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<uint8_t, 4, true, int>& sizes,
			  bool init_update, cudaStream_t stream){


      // threads 
      int maxThreads = 1;
      int dim = 1;
      int numThreads = std::min(dim, maxThreads);

      // blocks
      int numGrids = 1;

      // launch
      if (init_update){
	auto grid = dim3(numGrids);
	auto block = dim3(numThreads);
	init_centroids_kernel<T><<<grid,block,0,stream>>>(dists,burst,blocks,
							  centroids,clusters,sizes);
      }else{
	run_update_centroids(dists,burst,blocks,centroids,clusters,sizes,stream);
      }
      CUDA_TEST_ERROR();
							    
    }


    //
    // Template Inits
    //
    void update_centroids(Tensor<float, 5, true, int>& dists,
			  Tensor<float, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<float, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<uint8_t, 4, true, int>& sizes,
			  bool init_update, cudaStream_t stream){
      update_centroids<float>(dists,burst,blocks,centroids,
			      clusters,sizes,init_update,stream);
    }

    void update_centroids(Tensor<half, 5, true, int>& dists,
			  Tensor<half, 4, true, int>& burst,
			  Tensor<int, 5, true, int>& blocks,
			  Tensor<half, 5, true, int>& centroids,
			  Tensor<uint8_t, 4, true, int>& clusters,
			  Tensor<uint8_t, 4, true, int>& sizes,
			  bool init_update, cudaStream_t stream){
      update_centroids<half>(dists,burst,blocks,centroids,
			     clusters,sizes,init_update,stream);
    }


  }
}
