
"""
Compute the average of a subset of the burst
using (i) the original burst, (ii) indices,
and (iii) weights indicating the burst's subsets

"""


# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- kmb_search --
from ..utils import divUp

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

def compute_subset_ave(burst,indices,weights,ps):

    # -- shapes --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h_batch,w_batch = indices.shape

    # -- init ave --
    ave = torch.zeros(c,ps,ps,s,h_batch,w_batch).to(device)

    # -- run launcher --
    compute_subset_ave_launcher(ave,burst,indices,weights)

    # -- format from kernel --
    # none!

    return ave


def compute_subset_ave_launcher(ave,burst,indices,weights):

    # -- shapes --
    c,t,h,w = burst.shape
    two,t,s,h_batch,w_batch = indices.shape
    c,ps,ps,s,h_batch,w_batch = ave.shape

    # -- numbify the torch tensors --
    ave_nba = cuda.as_cuda_array(ave)
    burst_nba = cuda.as_cuda_array(burst)
    indices_nba = cuda.as_cuda_array(indices)
    weights_nba = cuda.as_cuda_array(weights)
    
    # -- tiling --
    hTile = 4
    wTile = 4
    sTile = 4

    # -- launch params --
    hBlocks = divUp(h_batch,hTile)
    wBlocks = divUp(w_batch,wTile)
    sBlocks = divUp(s,sTile)
    blocks = (hBlocks,wBlocks,sBlocks)
    threads = (c,ps,ps)

    # -- launch kernel --
    compute_subset_ave_kernel[blocks,threads](ave_nba,burst_nba,
                                              indices_nba,weights_nba,
                                              hTile,wTile,sTile)

@cuda.jit
def compute_subset_ave_kernel(ave,burst,indices,weights,hTile,wTile,sTile):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        return val
        
    # -- shapes --
    f,t,h,w = burst.shape
    f,ps,ps,s,h_batch,w_batch = ave.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hStart = hTile*cuda.blockIdx.x
    wStart = wTile*cuda.blockIdx.y
    sStart = sTile*cuda.blockIdx.z
    fi = cuda.threadIdx.x
    pi = cuda.threadIdx.y
    pj = cuda.threadIdx.z

    # -- compute dists --
    for hiter in range(hTile):
        hi = hStart + hiter
        if hi >= h: continue
        for witer in range(wTile):
            wi = wStart + witer
            if wi >= w: continue
            for siter in range(sTile):
                si = sStart + siter
                if si >= s: continue

                d_val,a_val,z_val = 0,0,0
                for ci in range(t):
        
                    w_val = weights[ci]#[si][hi][wi]
                    blkH = indices[0,ci,si,hi,wi]
                    blkW = indices[1,ci,si,hi,wi]
                    top,left = blkH-psHalf,blkW-psHalf
        
        
                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)
                    b_val = burst[fi][ci][bH][bW]
                    a_val += w_val*b_val
                    z_val += w_val
        
                ave[fi][pi][pj][si][hi][wi] = a_val/z_val
            
