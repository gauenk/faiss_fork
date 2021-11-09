
# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- imports --
from .cluster_update_impl import init_clusters
from .centroid_update_impl import fill_sframes_ecentroids
from .centroid_update_impl import update_ecentroids

def get_centroids_for_ave(ctype,noisy,clean,indices,ps):
    # -- get centroids --
    device = noisy.device
    c,nframes,h,w = clean.shape
    ref,nsearch = nframes//2,indices.shape[2]
    pimg = parse_ctype(ctype,noisy,clean)
    clusters,sizes = init_clusters(nframes,nframes,nsearch,h,w,device)
    centroids = update_ecentroids(pimg,indices,clusters,sizes,ps)
    if "clean" in ctype:# and "-" in ctype:
        centroids = torch.normal(centroids,std=100./(255.*5))
        # ntype = ctype.split("-")[1]
        # centroids = ref_noise(ntype,centroids)

    return centroids,clusters,sizes

def ref_noise(ntype,centroids):
    if ntype == "v1":
        centroids = torch.normal(centroids,std=100./(255*3))
    else:
        raise KeyError("Uknown noise type [{ntype}] for ref centroid.")
    return centroids
                
def parse_ctype(ctype,noisy,clean):
    cimg = None
    if "clean" in ctype: cimg = clean
    elif ctype == "noisy": cimg = noisy
    else: raise ValueError(f"unknown [centroid type] param [{ctype}]")
    return cimg
