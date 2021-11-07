
# -- python --
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- pytorch --
import torch

def init_centroids(burst,s,tK):
    # -- unpack --
    dtype = burst.type()
    device = burst.device
    c,t,h,w = burst.shape
    
    # -- numba --
    burst = burst.cpu().numpy()
    centroids = np.zeros((tK,c,s,h,w)).astype(np.float)
    # init_centroids_numba(burst,centroids)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)

    return centroids

@njit
def init_centroids_numba(burst,clusters,csizes,centroids):
    pass

def update_centroids(burst,blocks,clusters,csizes):

    # -- unpack --
    dtype = burst.type()
    device = burst.device
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK,s,h,w = csizes.shape
    
    # -- numba --
    burst = burst.cpu().numpy()
    blocks = blocks.cpu().numpy()
    clusters = clusters.cpu().numpy()
    csizes = csizes.cpu().numpy()
    centroids = np.zeros((c,tK,s,h,w)).astype(np.float)
    update_centroids_numba(burst,blocks,clusters,csizes,centroids)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)

    return centroids

@njit
def update_centroids_numba(burst,blocks,clusters,csizes,centroids):

    # -- shapes --
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK = centroids.shape[0]

    def bounds(p,lim):
        if p < 0: return -p
        elif p > lim: return 2*lim - p
        else: return p

    # -- assignment --
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                for ci in prange(c): # different features
                    for cid in prange(tK):
                        Z = csizes[cid,si,hi,wi]
                        if Z == 0: continue
                        for ti in range(t):
                            prop_cid = clusters[ti,si,hi,wi]
                            if prop_cid == cid:
                                bH = blocks[0,ti,si,hi,wi]
                                bW = blocks[1,ti,si,hi,wi]
                                bH = bounds(bH,h-1)
                                bW = bounds(bW,w-1)
                                b = burst[ci,ti,bH,bW]
                                centroids[ci,cid,si,hi,wi] += b/Z

# -------------------------------------------
#
#        Update Expanded Centroids
#
# -------------------------------------------

def update_ecentroids(burst,blocks,clusters,csizes,ps):

    # -- unpack --
    dtype = burst.type()
    device = burst.device
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK,s,h,w = csizes.shape
    # print("burst.shape: ",burst.shape)
    # print("blocks.shape: ",blocks.shape)
    # print("clusters.shape: ",clusters.shape)
    # print("sizes.shape: ",csizes.shape)
    # print(torch.any(torch.isnan(burst)))
    # print(burst.max().item(),burst.min().item())
    
    # -- numba --
    burst = burst.cpu().numpy()
    blocks = blocks.cpu().numpy()
    clusters = clusters.cpu().numpy()
    csizes = csizes.cpu().numpy()
    centroids = np.zeros((c,tK,s,h,w,ps,ps)).astype(np.float)
    update_ecentroids_numba(centroids,burst,blocks,clusters,csizes)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)

    return centroids

@njit
def update_ecentroids_numba(centroids,burst,blocks,clusters,csizes):

    # -- shapes --
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK = centroids.shape[1]
    ps = centroids.shape[-1]
    psHalf = ps//2

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- assignment --
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                for cid in range(tK):
                    Z = csizes[cid,si,hi,wi]
                    if Z == 0: continue
                    for ci in range(c): # different features
                        for ti in range(t):
                            prop_cid = clusters[ti,si,hi,wi]
                            if prop_cid == cid:
                                blkH = blocks[0,ti,si,hi,wi]
                                blkW = blocks[1,ti,si,hi,wi]
                                top,left = blkH-psHalf,blkW-psHalf
                                for pi in range(ps):
                                    for pj in range(ps):
                                        bH = bounds(top+pi,h-1)
                                        bW = bounds(left+pj,w-1)
                                        b = burst[ci,ti,bH,bW]
                                        centroids[ci,cid,si,hi,wi,pi,pj] += b/Z
                      
# ------------------------------------------------
#
#     Fill Expanded Centroids with Search Frames
#
# ------------------------------------------------

def fill_sframes_ecentroids(burst,indices,sframes,ps):
    """
    Fill a centroid with the values from burst
    """

    # -- unpack --
    dtype = burst.type()
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = indices.shape
    tK = len(sframes)

    # -- create outputs --
    centroids = np.zeros((c,tK,s,h,w,ps,ps)).astype(np.float)
    clusters = 255*np.ones((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)

    # -- numba --
    burst = burst.cpu().numpy()
    indices = indices.cpu().numpy()
    sframes = sframes.cpu().numpy()
    fill_ecentroids_numba(centroids,clusters,sizes,burst,indices,sframes)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)
    ave = torch.mean(centroids,dim=1)
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    # -- aug size to modify centroids --
    c,t,s,h,w,ps,ps = centroids.shape
    aug_sizes = repeat(sizes,'t s h w -> c t s h w p1 p2',c=c,p1=ps,p2=ps)
    centroids[torch.where(aug_sizes==0)]=float("nan")

    # -- ref centroid --
    ref_cid = (t//2)%tK
    ref_centroid = centroids[:,ref_cid]

    return centroids,clusters,sizes,ave,ref_centroid

# ------------------------------------------------
#
#     Fill Expanded Centroids when kmeansK == t
#
# ------------------------------------------------

def fill_ecentroids(burst,indices,ps,kmeansK):
    """
    Fill a centroid with the values from burst
    """

    # -- unpack --
    dtype = burst.type()
    device = burst.device
    two,t,s,h,w = indices.shape
    c,t,h,w = burst.shape
    tK = kmeansK
    sframes = np.arange(tk)

    # -- create outputs --
    centroids = np.zeros((c,tK,s,h,w,ps,ps)).astype(np.float)
    clusters = 255*np.ones((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)

    # -- numba --
    burst = burst.cpu().numpy()
    indices = indices.cpu().numpy()
    centroids = np.zeros((c,tK,s,h,w,ps,ps)).astype(np.float)
    assert t == tK, "num clusters must be eq to num frames for fill."
    fill_ecentroids_numba(centroids,clusters,sizes,burst,indices,sframes)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)
    ave = torch.mean(centroids,dim=1)
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    # -- ref centroid --
    ref_cid = (t//2)%tK
    ref_centroid = centroids[:,ref_cid]

    return centroids,clusters,sizes,ave,ref_centroid

@njit
def fill_ecentroids_numba(centroids,clusters,sizes,burst,indices,sframes):

    # -- shapes --
    c,t,h_b,w_b = burst.shape
    c,tK,s,h,w,ps,ps = centroids.shape
    tK = centroids.shape[1]
    assert tK == len(sframes),"search frames and num centroids must match."
    ps = centroids.shape[-1]
    psHalf = ps//2

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- assignment --
    for hi in range(h):
        for wi in range(w):
            for si in range(s):
                for ti in range(tK):
                    tj = sframes[ti]
                    for ci in range(c):
                        for pi in range(ps):
                            for pj in range(ps):
                                bH = indices[0,tj,si,hi,wi]
                                bW = indices[1,tj,si,hi,wi]
                                top,left = bH-psHalf,bW-psHalf
                                bH = bounds(top+pi,h_b-1)
                                bW = bounds(left+pj,w_b-1)
                                b = burst[ci,tj,bH,bW]
                                centroids[ci,ti,si,hi,wi,pi,pj] = b
                    clusters[tj][si][hi][wi] = ti
                    sizes[ti][si][hi][wi] += 1
