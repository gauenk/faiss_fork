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


def compute_burst_dists(centroids,sizes,ave,ps):

    # -- create output --
    device = centroids.device
    f,tK,s,h,w = centroids.shape
    vals = np.zeros((s,h,w)).astype(np.float32)

    # -- to numpy --
    centroids = centroids.cpu().numpy()
    sizes = sizes.cpu().numpy()
    ave = ave.cpu().numpy()

    # -- numba --
    compute_burst_dists_numba(vals,centroids,sizes,ave,ps)

    # -- to torch --
    vals = torch.FloatTensor(vals).to(device)

    return vals

@njit
def compute_burst_dists_numba(vals,centroids,sizes,ave,ps):
    def ps_select(start,limit,ps):
        sel = np.arange(start,start+ps)
        ltz = np.where(sel < 0)
        gtl = np.where(sel > limit)
        sel[ltz] = -sel[ltz]
        sel[gtl] = 2*limit - sel[gtl]
        return sel

    tK,h,w = vals.shape
    f,tK,s,h,w = centroids.shape
    Z = ps * ps 
    psHalf = ps//2

    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                num_ci = 0
                for ci in range(tK):
                    size = sizes[ci][si][hi][wi]
                    if size == 0: continue
                    num_ci += 1
                    for fi in range(f):
                        h_grid = ps_select(hi-psHalf,h-1,ps)
                        w_grid = ps_select(wi-psHalf,w-1,ps)
                        for pi in range(ps):
                            for pj in range(ps):
                                hp,wp = h_grid[pi],w_grid[pj]
                                cent = centroids[fi][ci][si][hp][wp]
                                ave_fi = ave[fi][si][hp][wp]
                                vals[si][hi][wi] += (cent - ave_fi)**2
                vals[si][hi][wi] /= num_ci

# ----------------------------------------------
#
#    Compute L2 v.s Ave
#    using Expanded Centroid
#    & Expanded Dists
#
# ----------------------------------------------

def compute_ecentroid_edists(centroids,sizes,ave,t):

    # -- create output --
    device = centroids.device
    f,tK,s,h,w,ps,ps = centroids.shape
    vals = np.zeros((tK,s,h,w)).astype(np.float32)

    # -- to numpy --
    centroids = centroids.cpu().numpy()
    sizes = sizes.cpu().numpy()
    ave = ave.cpu().numpy()

    # -- numba --
    compute_ecentroid_edists_numba(vals,centroids,sizes,ave,t)

    # -- to torch --
    vals = torch.FloatTensor(vals).to(device)

    return vals

@njit
def compute_ecentroid_edists_numba(vals,centroids,sizes,ave,t):
    """
    Expanded Centroids
    Expanded Dists
    """
    tK,s,h,w = vals.shape
    f,tK,s,h,w,ps,ps = centroids.shape
    eps,F_MAX = 1e-8,np.nan
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                # -- P*P*F*(C+1) --
                for ci in range(tK):
                    size = sizes[ci][si][hi][wi]
                    vsize = size > 0 and size <= t
                    for fi in range(f):
                        for pi in range(ps):
                            for pj in range(ps):
                                cent = centroids[fi][ci][si][hi][wi][pi][pj]
                                ave_f = ave[fi][si][hi][wi][pi][pj]
                                vals[ci][si][hi][wi] += (cent - ave_f)**2
                    val = vals[ci][si][hi][wi]#*size#/tK#/(size+eps)
                    val = val if vsize else F_MAX
                    vals[ci][si][hi][wi] = val

# ----------------------------------------------
#
#    Compute L2 v.s Ave using Expanded Centroid
#
# ----------------------------------------------

def compute_ecentroid_dists(centroids,sizes,ave):

    # -- create output --
    device = centroids.device
    f,tK,s,h,w,ps,ps = centroids.shape
    vals = np.zeros((s,h,w)).astype(np.float32)

    # -- to numpy --
    centroids = centroids.cpu().numpy()
    sizes = sizes.cpu().numpy()
    ave = ave.cpu().numpy()

    # -- numba --
    compute_ecentroid_dists_numba(vals,centroids,sizes,ave)

    # -- to torch --
    vals = torch.FloatTensor(vals).to(device)

    return vals

@njit
def compute_ecentroid_dists_numba(vals,centroids,sizes,ave):
    s,h,w = vals.shape
    f,tK,s,h,w,ps,ps = centroids.shape
    F_MAX = 10000.
    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                # -- P*P*F*(C+1) --
                num_ci = 0
                for ci in range(tK):
                    size = sizes[ci][si][hi][wi]
                    if size == 0: continue
                    num_ci += 1
                    for fi in range(f):
                        for pi in range(ps):
                            for pj in range(ps):
                                cent = centroids[fi][ci][si][hi][wi][pi][pj]
                                ave_f = ave[fi][si][hi][wi][pi][pj]
                                vals[si][hi][wi] += (cent - ave_f)**2
                val = vals[si][hi][wi]
                val = val / num_ci if num_ci > 1 else F_MAX
                vals[si][hi][wi] = val

@njit
def compute_ecent_dists_numba_v2(vals,ecent,sizes,agg,ps):
    tK,h,w = vals.shape
    two,t,s,h,w = indices.shape
    f,tK,ps2,h,w = ecent.shape

    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                # -- P*P*F*(C+1) --

                # -> burst(3d) and indices(5d) to pcents (5d) [f,c,s,p,p]
                # -> this has to be done in the clustering too...?
                # -> ?

                # -- compute cluster means --
                for ti in range(t):
                    ci = clusters[ti][hi][wi]
                    for fi in range(f):
                        agg[fi][ci][hi][wi] += burst[fi][hi][wi]

                # -- compute ave --
                for fi in range(f):
                    for ci in range(tK):
                        ave[fi][hi][wi] += agg[fi][ci][hi][wi]
                    ave[fi][hi][wi] /= tK

                # -- compute delta to ave --
                for fi in range(f):
                    for ci in range(tK):
                        vals[fi][hi][wi] += (ave[fi][hi][wi] - agg[fi][ci][hi][wi])**2
                
