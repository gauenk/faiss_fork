
# -- python --
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- pytorch --
import torch

# -- local --
from .sup_cluster import sup_clusters_numba_v1,sup_clusters_numba_v2

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#              Supervised Clustering
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def sup_clusters(clean,indices,indices_gt,sframes,ps,version="v1"):
    
    # -- init vars --
    device = indices.device
    two,t,s,h,w = indices.shape
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((t,s,h,w)).astype(np.uint8)

    # -- launch numba --
    clean = clean.cpu().numpy()
    indices = indices.cpu().numpy()
    indices_gt = indices_gt.cpu().numpy()
    sframes = sframes.cpu().numpy()
    # print("pre numba: ",sframes)
    if version == "v1":
        sup_clusters_numba_v1(clusters,sizes,clean,indices,indices_gt,sframes,ps)
    elif version == "v2":
        sup_clusters_numba_v2(clusters,sizes,clean,indices,indices_gt,sframes,ps)
    else:
        raise ValueError(f"Uknown cluster version [{version}]")

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#              Cluster Update using Pairwise Distances
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def update_clusters(dists):

    # -- unpack --
    device = dists.device
    t,tK,s,h,w = dists.shape
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    dists = dists.cpu().numpy()
    update_clusters_numba(dists,clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def update_clusters_numba(dists,clusters,sizes):
    t,tK,s,h,w = dists.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t0_argmin = -1
                    for t1 in range(tK):
                        d = dists[t0,t1,si,hi,wi]
                        if d < dmin:
                            dmin = d
                            t0_argmin = t1
                    clusters[t0,si,hi,wi] = t0_argmin
                    sizes[t0_argmin,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Init Cluster Update
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def init_clusters(t,tK,s,h,w,device='cuda:0'):

    # -- unpack --
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    init_clusters_numba(clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def init_clusters_numba(clusters,sizes):
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t1 = t0 % tK
                    clusters[t0,si,hi,wi] = t1
                    sizes[t1,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Randomly Init Clusters
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def rand_clusters(t,tK,s,h,w,device='cuda:0'):

    # -- unpack --
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    rand_clusters_numba(clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def rand_clusters_numba(clusters,sizes):
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t1 = int(np.random.rand(1)[0]*tK)
                    clusters[t0,si,hi,wi] = t1
                    sizes[t1,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Self Similar Cluster Update
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def update_self_clusters(dists):

    # dists from self_pairwise_distance

    # -- unpack --
    device = dists.device
    t,t,s,h,w = dists.shape
    clusters = np.zeros((t,s,h,w)).astype(np.int)
    
    # -- numba --
    dists = dists.cpu().numpy()
    update_self_clusters_numba(dists,clusters)

    # -- to torch --
    clusters = torch.IntTensor(clusters).to(device)

    return clusters

@njit
def update_self_clusters_numba(dists,clusters):

    def reflect_pair(t0,t1):
        if t1 > t0:
            r0 = t1
            r1 = t0
        else:
            r0 = t0
            r1 = t1
        return r0,r1

    t,t,s,h,w = dists.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t0_argmin = -1
                    for t1 in range(t):
                        if t0 == t1: continue
                        r0,r1 = reflect_pair(t0,t1)
                        d = dists[r0,r1,si,hi,wi]
                        if d < dmin:
                            dmin = d
                            t0_argmin = t1
                    clusters[t0,si,hi,wi] = t0_argmin

