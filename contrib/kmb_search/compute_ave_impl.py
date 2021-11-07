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
from numba import jit,njit,prange

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

# ----------------------------------------------
#
#    Compute Ave over Expanded Centroid
#
# ----------------------------------------------

def compute_ecentroid_ave(centroids,sizes):

    # -- create output --
    device = centroids.device
    f,tK,s,h,w,ps,ps = centroids.shape

    # -- to numpy --
    centroids = centroids.cpu().numpy()
    sizes = sizes.cpu().numpy()
    ave = np.zeros((f,s,h,w,ps,ps))

    # -- numba --
    compute_ecentroid_ave_numba(centroids,sizes,ave)

    # -- to torch --
    ave = torch.FloatTensor(ave).to(device)

    return ave

@njit
def compute_ecentroid_ave_numba(centroids,sizes,ave):
    f,tK,s,h,w,ps,ps = centroids.shape
    F_MAX = 10000.
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                num_ci = 0
                for ci in range(tK):
                    size = sizes[ci][si][hi][wi]
                    if size == 0: continue
                    num_ci += 1
                    for fi in range(f):
                        for pi in range(ps):
                            for pj in range(ps):
                                cent = centroids[fi][ci][si][hi][wi][pi][pj]*size/tK
                                ave[fi][si][hi][wi][pi][pj] += cent
                for fi in range(f):
                    for pi in range(ps):
                        for pj in range(ps):
                            val = ave[fi][si][hi][wi][pi][pj]
                            val = val / num_ci if num_ci > 0 else F_MAX
                            ave[fi][si][hi][wi][pi][pj] = val

# ----------------------------------------------
#
#    Compute Ave over Standard Centroid
#
# ----------------------------------------------

def compute_centroid_ave(centroids,sizes):

    # -- create output --
    device = centroids.device
    f,tK,s,h,w = centroids.shape
    vals = np.zeros((s,h,w)).astype(np.float32)

    # -- to numpy --
    centroids = centroids.cpu().numpy()
    sizes = sizes.cpu().numpy()
    ave = np.zeros((f,s,h,w))

    # -- numba --
    compute_centroid_ave_numba(centroids,sizes,ave)

    # -- to torch --
    ave = torch.FloatTensor(ave).to(device)

    return ave

@njit
def compute_centroid_ave_numba(centroids,sizes,ave):
    f,tK,s,h,w = centroids.shape
    F_MAX = 10000.
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                num_ci = 0
                for ci in range(tK):
                    size = sizes[ci][si][hi][wi]
                    if size == 0: continue
                    num_ci += 1
                    for fi in range(f):
                        cent = centroids[fi][ci][si][hi][wi]
                        ave[fi][si][hi][wi] += cent
                for fi in range(f):
                    val = ave[fi][si][hi][wi]
                    ave[fi][si][hi][wi] = val / num_ci if num_ci > 0 else F_MAX
