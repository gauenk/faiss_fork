

# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda


# ------------------------------------------
#
#           Init Clusters
#
# ------------------------------------------

def init_clusters(nframes,kmeansK,nsearch,h,w,device):

    # -- create outputs --
    clusters = torch.zeros((nframes,nsearch,h,w)).type(torch.byte)
    sizes = torch.zeros((kmeansK,nsearch,h,w)).type(torch.byte)
    clusters.to(device)
    sizes.to(device)

    # -- run launcher --
    init_clusters_launcher(clusters,sizes,nframes)

    return clusters,sizes


def init_clusters_launcher(clusters,sizes,nframes):

    # -- shapes --
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]

    # -- numbify the torch tensors --
    clusters_nba = cuda.as_cuda_array(clusters)
    sizes_nba = cuda.as_cuda_array(sizes)

    # -- launch params --
    threads = s
    blocks = (h_batch,w_batch)

    # -- launch kernel --
    init_clusters_kernel[blocks,threads](clusters_nba,sizes_nba)

@cuda
def init_clusters_kernel(clusters,sizes):
    
    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- shapes --
    nframes = clusters.shape[0]
    nclusters = sizes.shape[0]

    # -- assign --
    for t in range(nframes):
        clusters[t][si][hi][wi] = t % nclusters
        cid = clusters[t][si][hi][wi]
        sizes[ci][si][hi][wi] += 1


# ------------------------------------------
#
#           Update Clusters
#
# ------------------------------------------

def update_clusters(dists):

    # -- unpack --
    device = dists.device
    t,tK,s,h,w = dists.shape
    
    # -- create output --
    clusters = torch.zeros((t,s,h,w)).type(torch.byte)
    sizes = torch.zeros(tK,h,w).type(torch.byte)
    clusters = clusters.to(device)
    dists = dists.to(device)

    # -- run launcher --
    update_clusters_launcher(clusters,sizes,dists)


    return clusters,sizes

def update_clusters_launcher(clusters,sizes,dists):

    # -- shapes --
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]

    # -- numbify the torch tensors --
    clusters_nba = cuda.as_cuda_array(clusters)
    sizes_nba = cuda.as_cuda_array(sizes)
    dists_nba = cuda.as_cuda_array(dists)

    # -- launch params --
    threads = s
    blocks = (h_batch,w_batch)

    # -- launch kernel --
    update_clusters_kernel[blocks,threads](clusters_nba,sizes_nba,dists_nba)

@cuda
def update_clusters_kernel(clusters,sizes,dists):

    # -- shapes --
    nframes,nclusters,s,h,w = dists.shape

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y
    si = cuda.threadIdx.x

    # -- assign --
    for ti in range(nframes):
        cmin,val = 0,1000.
        for ci in range(nclusters):
            dval = dists[ti][ci][si][hi][wi]
            if dval < val:
                cmin,val = ci,dval
        clusters[ti][si][hi][wi] = cmin
        sizes[cmin][si][hi][wi] += 1

    
    
