

# -- python --
import numpy as np
from einops import rearrange,repeat

# -- python --
from numba import jit,njit,prange,cuda

# -- pytorch --
import torch



def update_centroids():

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
    update_centroids_launcher(burst,blocks,clusters,csizes,centroids)

    # -- to torch --
    centroids = torch.FloatTensor(centroids)
    centroids = centroids.type(dtype).to(device)

    return centroids

def update_centroids_launcher(burst,blocks,clusters,csizes,centroids):
    pass


@cuda
def update_centroids_numba(burst,blocks,clusters,csizes,centroids):
    pass
