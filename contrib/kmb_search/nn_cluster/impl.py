
import torch
import numpy as np
from einops import rearrange,repeat


from .cluster_update_impl import init_clusters
from .centroid_update_impl import update_ecentroids

def run_nn_cluster(burst,indices,ps,model):
    """

    Interface with cluster parser for kmb search.

    """

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = indices.shape

    # -- create centroids for each frame --
    clusters,sizes = init_clusters(t,t,s,h,w,device)
    eburst = update_ecentroids(burst,indices,clusters,sizes,ps)
    
    # -- shape --
    _eburst = rearrange(eburst,'c t s h w p1 p2 -> (s h w) t c p1 p2')
    clusters = model(_eburst)
    clusters = rearrange(clusters,'(s h w) -> s h w',s=s,h=h,w=w)

    # -- comp. sizes --
    ones = torch.ones_like(clusters)
    sizes = torch.scatter_add(ones,dim=0,index=clusters)

    return None,centroids,clusters,sizes
