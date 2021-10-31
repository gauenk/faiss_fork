
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/KmBurstAve4d.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <algorithm>

namespace faiss {
  namespace gpu {


    /*********************************************

           Compute Ave Along 1st dim

    *********************************************/

    template <typename T>
    __global__ void kmb_ave_kernel4d(Tensor<T, 4, true, int> tensor,
				     Tensor<T, 3, true, int> ave){

      // height,width
      int h = blockIdx.x;
      int w = blockIdx.y;

      // get dims to comp indices from thread
      int nframes = tensor.getSize(0);
      int nblocks = tensor.getSize(1);
      int dim = nblocks;
      T inv_nframes = 1./nframes;

      // helpers
      int fIdx,b;
      T ave_val;

      // set clusters
      for (int tIdx = threadIdx.x; tIdx < dim; tIdx += blockDim.x){
	b = tIdx % nblocks;

	ave_val = 0;
	for (int fIdx = 0; fIdx < nframes; ++fIdx){
	  ave_val += tensor[fIdx][b][h][w];
	}
	ave[b][h][w] = Math<T>::mul(ave_val,inv_nframes);

      }
    }

    template <typename T>
    void kmb_ave4d(Tensor<T, 4, true, int> tensor,
		   Tensor<T, 3, true, int> ave,
		   cudaStream_t stream){

      // shapes
      int nframes = tensor.getSize(0);
      int bBatch = tensor.getSize(1);
      int hBatch = tensor.getSize(2);
      int wBatch = tensor.getSize(3);
      
      // threads 
      int maxThreads = (int)getMaxThreadsCurrentDevice();
      int dim = bBatch;
      int numThreads = std::min(dim, maxThreads);

      // launch
      auto grid = dim3(hBatch,wBatch);
      auto block = dim3(numThreads);

      // launch kernel
      kmb_ave_kernel4d<<<grid,block,0,stream>>>(tensor,ave);

      // error check
      CUDA_TEST_ERROR();
    }

    void kmb_ave4d(Tensor<float, 4, true, int> tensor,
		   Tensor<float, 3, true, int> ave,
		   cudaStream_t stream){
      kmb_ave4d<float>(tensor,ave,stream);
    }

    void kmb_ave4d(Tensor<half, 4, true, int> tensor,
		   Tensor<half, 3, true, int> ave,
		   cudaStream_t stream){
      kmb_ave4d<half>(tensor,ave,stream);
    }

  } // namespace gpu
} // namespace faiss
