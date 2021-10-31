
#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
  namespace gpu {

    /*********************************************

            Compute Mode Pairs

    *********************************************/

    __inline__
    float compute_mode_pairs(float std, int patchsize, int nftrs){
      int P = patchsize * patchsize * nftrs;
      float p_float = (float)P;
      float p_ratio = (p_float-2)/p_float;
      float mode = 2 * p_ratio * std * std;
      return mode;
    }

    /*********************************************

            Compute Mode Burst

    *********************************************/

    __inline__
    float compute_mode_burst(float std, int patchsize, int nftrs, int t){
      int P = patchsize * patchsize * nftrs;
      float p_ratio = (P-2)/(float)P;
      int t2 = t * t;
      // ( (t-1)/t )**2 + (t-1)/t**2
      float t_ratio = ((t-1)/(float)t);
      t_ratio = t_ratio * t_ratio;
      t_ratio = t_ratio + (t-1)/(float)t2;
      float var = std * std;
      float mode = p_ratio * t_ratio * var;
      return mode;
    }


    /*********************************************

           Compute Mode Centroids

    *********************************************/

    void compute_mode_centroids(float std, int patchsize, int nftrs,
				Tensor<uint8_t, 4, true, int> sizes,
				Tensor<float, 4, true, int> modes,
				cudaStream_t stream);

    void compute_mode_centroids(float std, int patchsize, int nftrs,
				Tensor<uint8_t, 4, true, int> sizes,
				Tensor<half, 4, true, int> modes,
				cudaStream_t stream);

  }
}
