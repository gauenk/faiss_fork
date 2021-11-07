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

# -- local imports --
from .centroid_impl import 


import torch

from .centroid_update_impl import update_centroids
from .cluster_update_impl import update_clusters,init_clusters
from .pwd_impl import compute_pairwise_distance

def run_kmeans(burst,indices,kmeansK,ps,offset,niters=10):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,nsearch,h,w = indices.shape

    # -- init alg --
    clusters,sizes = init_clusters(t,kmeansK,nsearch,h,w,device)
    # print("[kmeans.burst] max,min: ",burst.max().item(),burst.min().item())
    cnorms = compute_cnorm(burst,indices,clusters,sizes,ps)
    # centroids = update_centroids(burst,indices,clusters,sizes,ps)
    # print("[kmeans.centroids] max,min: ",centroids.max().item(),centroids.min().item())
    # assert torch.all(sizes == 1).item() == True,"all ones."
    # ecentroids,_,_ = fill_ecentroids(burst,indices,ps,kmeansK)
    # delta = torch.sum(torch.abs(centroids-ecentroids)).item()
    # assert delta < 1e-8,"delta must be small here."

    # -- run loop --
    for iter_i in range(niters):
        
        # -- compute pwd --
        dists = compute_pairwise_distance(burst,indices,centroids,offset)
        
        # -- update clusters --
        clusters,sizes = update_clusters(dists)
        # assert torch.all(sizes == 1).item() == True,"all ones."

        # -- update centroids --
        centroids = update_ecentroids(burst,indices,clusters,sizes,ps)

    return dists,clusters,sizes,centroids

