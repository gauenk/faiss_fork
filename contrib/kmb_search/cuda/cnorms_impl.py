"""

Compute the differences for each block
using the centroids

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

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad


def compute_cnorm(burst,indices,clusters,sizes,ps):

    # -- create output --
    device = burst.device
    f,t,h,w = burst.shape
    two,t,s,h_batch,w_batch = indices.shape
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]

    # -- init dists --
    cnorms = torch.zeros(tK,s,h_batch,w_batch).to(device)

    # -- run launcher --
    compute_cnorm_launcher(cnorms,burst,indices,clusters,sizes,ps)

    # -- format dists --
    # none!

    return dists

def compute_cnorm_launcher(cnorms,burst,indices,clusters,sizes,ps):

    # -- shapes --
    tK,s,h_batch,w_batch = cnorms.shape
    c,t,h,w = burst.shape
    two,t,s,h_batch,w_batch = indices.shape

    # -- numbify the torch tensors --
    cnorms_nba = cuda.as_cuda_array(cnorms)
    burst_nba = cuda.as_cuda_array(burst)
    indices_nba = cuda.as_cuda_array(indices)
    clusters_nba = cuda.as_cuda_array(clusters)
    sizes_nba = cuda.as_cuda_array(sizes)

    # -- launch params --
    threads = s
    blocks = (h_batch,w_batch)

    # -- launch kernel --
    compute_cnorm_kernel[blocks,threads](cnorms_nba,burst_nba,indices_nba,
                                         clusters_nba,sizes_nba,ps)

@cuda.jit
def compute_cnorm_kernel(cnorms,burst,indices,clusters,sizes,ps):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        # val = val % lim
        return val
        
    # -- shapes --
    f,t,h,w = burst.shape
    tK,s,h_batch,w_batch = cnorms.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- compute dists --
    z_val,d_val = 0,0
    for ti in range(t):

        # -- frame weight --
        cid = clusters[ti][si][hi][wi]

        # -- get indices --
        blkH = indices[0,ti,si,hi,wi]
        blkW = indices[1,ti,si,hi,wi]

        # -- compute over features --
        for pi in range(ps):
            for pj in range(ps):
                for fi in range(f):

                    # -- inside entire image --
                    top,left = blkH-psHalf,blkW-psHalf
                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)

                    # -- get data --
                    b_val = burst[fi][ti][bH][bW]

                    # -- compute dist --
                    cnorms[cid][si][hi][wi] += b_val

    # -- normalize --
    for ti in range(t):
        cid = clusters[ti][si][hi][wi]
        size = sizes[cid][si][hi][wi]
        cnorms[cid][si][hi][wi] /= size
