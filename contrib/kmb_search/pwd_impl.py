# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                (Burst,ECentroid) Pairwise Distance (expanded centroid)
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def compute_epairwise_distance(burst,blocks,centroids,offset):
    # -- shapes and init --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    c,tK,s,h,w,ps,ps = centroids.shape
    dists = torch.zeros(t,tK,s,h,w).to(burst.device)
    assert two == 2,"blocks first dim must be 2."
    assert tK <= t,"Num of Clusters less than Num of Samples."

    # -- to numpy --
    dists = dists.cpu().numpy()
    burst = burst.cpu().numpy()
    blocks = blocks.cpu().numpy()
    centroids = centroids.cpu().numpy()

    # -- run numba --
    compute_epairwise_distance_numba(dists,burst,blocks,centroids,offset)

    # -- to torch --
    dists = torch.FloatTensor(dists).to(device)
    
    return dists

@njit
def compute_epairwise_distance_numba(dists,burst,blocks,centroids,offset):

    def ps_select(start,limit,ps):
        sel = np.arange(start,start+ps)
        ltz = np.where(sel < 0)
        gtl = np.where(sel > limit)
        sel[ltz] = -sel[ltz]
        sel[gtl] = 2*limit - sel[gtl]
        return sel

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- compute differences --
    eps = 1e-8 # breaking ties by choosing diag
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    c,tK,s,h,w,ps,ps = centroids.shape
    psHalf = ps//2
    Z = ps*ps*c
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    for t1 in prange(tK):
                        h_t0_s = blocks[0,t0,si,hi,wi] - psHalf
                        w_t0_s = blocks[1,t0,si,hi,wi] - psHalf
                        h_t1_s = hi - psHalf
                        w_t1_s = wi - psHalf
                        
                        h_t0 = ps_select(h_t0_s,h-1,ps)
                        w_t0 = ps_select(w_t0_s,w-1,ps)
                        h_t1 = ps_select(h_t1_s,h-1,ps)
                        w_t1 = ps_select(w_t1_s,w-1,ps)

                        blkH = blocks[0,t0,si,hi,wi]
                        blkW = blocks[1,t0,si,hi,wi]
                        top,left = blkH-psHalf,blkW-psHalf
                        for ci in range(c):
                            for pH in range(ps):
                                for pW in range(ps):
                                    bH = bounds(top+pH,h-1)
                                    bW = bounds(left+pW,w-1)
                                    b_t0 = burst[ci,t0,bH,bW]
                                    # b_t0 = burst[ci,t0,h_t0[pH],w_t0[pW]]
                                    b_t1 = centroids[ci,t1,si,hi,wi,pH,pW]
                                    diff = (b_t0 - b_t1)**2
                                    dists[t0,t1,si,hi,wi] += diff
                        dists[t0,t1,si,hi,wi] /= Z
                        dists[t0,t1,si,hi,wi] -= offset
                        dists[t0,t1,si,hi,wi] = abs(dists[t0,t1,si,hi,wi])
                        if t0 != t1: dists[t0,t1,si,hi,wi] += eps
    return dists


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                (Burst,Centroid) Pairwise Distance
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def compute_pairwise_distance(burst,blocks,centroids,ps,offset):
    # -- shapes and init --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    c,tK,s,h,w = centroids.shape
    dists = torch.zeros(t,tK,s,h,w).to(burst.device)
    assert two == 2,"blocks first dim must be 2."
    assert tK <= t,"Num of Clusters less than Num of Samples."

    # -- to numpy --
    dists = dists.cpu().numpy()
    burst = burst.cpu().numpy()
    blocks = blocks.cpu().numpy()
    centroids = centroids.cpu().numpy()

    # -- run numba --
    compute_pairwise_distance_numba(dists,burst,blocks,centroids,ps,offset)

    # -- to torch --
    dists = torch.FloatTensor(dists).to(device)
    
    return dists

@njit
def compute_pairwise_distance_numba(dists,burst,blocks,centroids,ps,offset):

    def ps_select(start,limit,ps):
        sel = np.arange(start,start+ps)
        ltz = np.where(sel < 0)
        gtl = np.where(sel > limit)
        sel[ltz] = -sel[ltz]
        sel[gtl] = 2*limit - sel[gtl]
        return sel

    # -- compute differences --
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    c,tK,s,h,w = centroids.shape
    psHalf = ps//2
    Z = ps*ps*c
    for t0 in prange(t):
        for t1 in prange(tK):
            for si in prange(s):
                for hi in prange(h):
                    for wi in prange(w):
                        h_t0_s = blocks[0,t0,si,hi,wi] - psHalf
                        w_t0_s = blocks[1,t0,si,hi,wi] - psHalf
                        h_t1_s = hi - psHalf
                        w_t1_s = wi - psHalf
                        
                        h_t0 = ps_select(h_t0_s,h-1,ps)
                        w_t0 = ps_select(w_t0_s,w-1,ps)
                        h_t1 = ps_select(h_t1_s,h-1,ps)
                        w_t1 = ps_select(w_t1_s,w-1,ps)
                        for ci in prange(c):
                            for pH in range(ps):
                                for pW in range(ps):
                                    b_t0 = burst[ci,t0,h_t0[pH],w_t0[pW]]
                                    b_t1 = centroids[ci,t1,si,h_t1[pH],w_t1[pW]]
                                    diff = (b_t0 - b_t1)**2
                                    dists[t0,t1,si,hi,wi] += diff
                        dists[t0,t1,si,hi,wi] /= Z
                        dists[t0,t1,si,hi,wi] -= offset
                        dists[t0,t1,si,hi,wi] = abs(dists[t0,t1,si,hi,wi])
    return dists

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Self Similar Pairwise Distance
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def compute_self_pairwise_distance(burst,blocks,ps):

    # -- shapes and init --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    dists = torch.zeros(t,t,s,h,w).to(burst.device)


    # -- to numpy --
    dists = dists.cpu().numpy()
    burst = burst.cpu().numpy()
    blocks = blocks.cpu().numpy()

    # -- run numba --
    compute_self_pairwise_distance_numba(dists,burst,blocks,ps)

    # -- normalize --
    Z = ps*ps*c
    dists = dists/Z

    # -- to torch --
    dists = torch.FloatTensor(dists).to(device)
    
    return dists

@njit
def compute_self_pairwise_distance_numba(dists,burst,blocks,ps):

    def ps_select(start,limit,ps):
        sel = np.arange(start,start+ps)
        ltz = np.where(sel < 0)
        gtl = np.where(sel > limit)
        sel[ltz] = -sel[ltz]
        sel[gtl] = 2*limit - sel[gtl]
        return sel

    # -- compute differences --
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    psHalf = ps//2
    for t0 in prange(t):
        for t1 in prange(t):
            if t1 >= t0: continue
            for si in prange(s):
                for hi in prange(h):
                    for wi in prange(w):
                        h_t0_s = blocks[0,t0,si,hi,wi] - psHalf
                        w_t0_s = blocks[1,t0,si,hi,wi] - psHalf
                        h_t1_s = blocks[0,t1,si,hi,wi] - psHalf
                        w_t1_s = blocks[1,t1,si,hi,wi] - psHalf
                        
                        h_t0 = ps_select(h_t0_s,h-1,ps)
                        w_t0 = ps_select(w_t0_s,w-1,ps)
                        h_t1 = ps_select(h_t1_s,h-1,ps)
                        w_t1 = ps_select(w_t1_s,w-1,ps)
                        for ci in prange(c):
                            for pH in range(ps):
                                for pW in range(ps):
                                    b_t0 = burst[ci,t0,h_t0[pH],w_t0[pW]]
                                    b_t1 = burst[ci,t1,h_t1[pH],w_t1[pW]]
                                    diff = (b_t0 - b_t1)**2
                                    dists[t0,t1,si,hi,wi] += diff
    return dists
