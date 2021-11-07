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


def compute_l2norm_cuda(burst,indices,weights,ave):

    # -- create output --
    device = burst.device
    f,t,h,w = burst.shape
    two,t,s,h_batch,w_batch = indices.shape

    # -- init dists --
    dists = torch.zeros(s,h_batch,w_batch).to(device)

    # -- run launcher --
    compute_l2norm_launcher(dists,burst,indices,weights,ave)

    # -- format dists --
    # none!

    return dists

def compute_l2norm_launcher(dists,burst,indices,weights,ave):

    # -- shapes --
    c,t,h,w = burst.shape
    f,ps,ps,s,h,w = ave.shape
    two,t,s,h_batch,w_batch = indices.shape

    # -- numbify the torch tensors --
    dists_nba = cuda.as_cuda_array(dists)
    burst_nba = cuda.as_cuda_array(burst)
    indices_nba = cuda.as_cuda_array(indices)
    weights_nba = cuda.as_cuda_array(weights)
    ave_nba = cuda.as_cuda_array(ave)

    # -- launch params --
    threads = s
    blocks = (h_batch,w_batch)

    # -- launch kernel --
    compute_l2norm_kernel[blocks,threads](dists_nba,burst_nba,
                                               indices_nba,weights_nba,ave_nba)

@cuda.jit
def compute_l2norm_kernel_v1(dists,burst,indices,weights,ave):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        # val = val % lim
        return val
        
    # -- shapes --
    f,t,h,w = burst.shape
    f,ps,ps,s,h_batch,w_batch = ave.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- compute dists --
    z_val,d_val = 0,0
    for ti in range(t):

        # -- frame weight --
        w_val = weights[ti]
        z_val += w_val

        # -- compute over features --
        for fi in range(f):
            for pi in range(ps):
                for pj in range(ps):


                    # -- get indices --
                    blkH = indices[0,ti,si,hi,wi]
                    blkW = indices[1,ti,si,hi,wi]

                    # -- inside entire image --
                    top,left = blkH-psHalf,blkW-psHalf
                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)

                    # -- get data --
                    b_val = burst[fi][ti][bH][bW]
                    a_val = ave[fi][pi][pj][si][hi][wi]

                    # -- compute dist --
                    d_val += w_val*(b_val - a_val)**2

    # -- dists --
    dists[si][hi][wi] = d_val/z_val

@cuda.jit
def compute_l2norm_kernel(dists,burst,indices,weights,ave):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        # val = val % lim
        return val
        
    # -- shapes --
    f,t,h,w = burst.shape
    f,ps,ps,s,h_batch,w_batch = ave.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- compute dists --
    for fi in range(f):
        for pi in range(ps):
            for pj in range(ps):
                d_val,d2_val,z_val = 0,0,0
                for ti in range(t):

                    # -- frame weight --
                    w_val = weights[ti]
                    z_val += w_val

                    # -- get indices --
                    blkH = indices[0,ti,si,hi,wi]
                    blkW = indices[1,ti,si,hi,wi]

                    # -- inside entire image --
                    top,left = blkH-psHalf,blkW-psHalf
                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)

                    # -- compute distances --
                    b_val = burst[fi][ti][bH][bW]
                    bw_val = w_val*b_val
                    d_val += bw_val
                    d2_val += bw_val*b_val

                # -- compute y^2 --
                a_val = ave[fi][pi][pj][si][hi][wi]
                a2_val = a_val*a_val

                # -- normalize --
                _dist = d2_val - a_val * d_val# / z_val
                # _dist = - a_val * d_val
                # _dist = _dist + a2_val

                # -- compute dist --
                dists[si][hi][wi] += _dist

@cuda.jit
def compute_l2norm_kernel_wimg(dists,burst,indices,weights,ave):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        # val = val % lim
        return val
        
    # -- shapes --
    f,t,h,w = burst.shape
    f,ps,ps,s,h_batch,w_batch = ave.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- compute dists --
    for fi in range(f):
        for pi in range(ps):
            for pj in range(ps):
                d_val,d2_val,z_val = 0,0,0
                for ti in range(t):

                    # -- frame weight --
                    w_val = weights[ti][si][hi][wi]
                    z_val += w_val

                    # -- get indices --
                    blkH = indices[0,ti,si,hi,wi]
                    blkW = indices[1,ti,si,hi,wi]

                    # -- inside entire image --
                    top,left = blkH-psHalf,blkW-psHalf
                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)

                    # -- compute distances --
                    b_val = burst[fi][ti][bH][bW]
                    bw_val = w_val*b_val
                    d_val += bw_val
                    d2_val += bw_val*b_val

                # -- compute y^2 --
                a_val = ave[fi][pi][pj][si][hi][wi]
                a2_val = a_val*a_val

                # -- normalize --
                _dist = d2_val - a_val * d_val# / z_val
                # _dist = - a_val * d_val
                # _dist = _dist + a2_val

                # -- compute dist --
                dists[si][hi][wi] += _dist

1
