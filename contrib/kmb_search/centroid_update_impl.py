
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
