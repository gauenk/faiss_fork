
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

def update_centroids(burst,clusters,csizes):

    # -- unpack --
    dtype = burst.type()
    device = burst.device
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK = csizes.shape
    
    # -- numba --
    burst = burst.cpu().numpy()
    clusters = clusters.cpu().numpy()
    csizes = csizes.cpu().numpy()
    centoids = np.zeros((tK,c,s,h,w)).astype(np.float)
    update_centroids_numba(burst,clusters,csizes,centroids)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)

    return centroids

@njit
def update_centroids_numba(burst,clusters,csizes,centroids):

    # -- shapes --
    c,t,h,w = burst.shape
    t,s,h,w = clusters.shape
    tK = centroids.shape[0]

    # -- assignment --
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                for ti in prange(t):
                    cid = clusters[ti,si,hi,wi]
                    for ci in prange(c):
                        centoids[cid,ci,si,hi,wi] += burst[ci,ti,hi,wi]/csizes[cid]
