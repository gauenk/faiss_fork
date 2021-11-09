
# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# --------------------------------------
#
#         Fill Ref L2 Vals
#
# --------------------------------------

def fill_ref_l2vals(val,clusters,l2_vals,ref=None):
    # ref l2_vals with the same underlying memory
    t = clusters.shape[0]
    tK,s,h,w = l2_vals.shape
    if ref is None: ref = t//2

    # l2vals[:,inds[0],inds[1],inds[2],inds[3],:,:] = val
    device = l2_vals.device
    l2_vals_nba = l2_vals.cpu().numpy()
    clusters_nba = clusters.cpu().numpy()
    fill_ref_l2vals_numba(val,clusters_nba,l2_vals_nba,ref)
    l2_vals_nba = torch.FloatTensor(l2_vals_nba).to(device)
    l2_vals[...] = l2_vals_nba

@njit
def fill_ref_l2vals_numba(val,clusters,l2_vals,ref):
    t = clusters.shape[0]
    tK,s,h,w = l2_vals.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                cid = clusters[ref,si,hi,wi].item()
                l2_vals[cid,si,hi,wi] = val

# --------------------------------------
#
#         Fill Ref Centroids
#
# --------------------------------------

def fill_ref_centroids(val,clusters,centroids,inds,ref=None):
    # ref centroids with the same underlying memory
    t = clusters.shape[0]
    c,tK,s,h,w,ps,ps = centroids.shape
    if ref is None: ref = t//2
    # inds = (clusters[ref] == clusters)
    # inds = torch.where(inds)
    # ref_centroids = torch.zeros_like(centroids)
    # print(inds)
    # print(len(inds))
    # centroids[:,inds[0],inds[1],inds[2],inds[3],:,:] = val
    device = centroids.device
    centroids_nba = centroids.cpu().numpy()
    clusters_nba = clusters.cpu().numpy()
    fill_ref_centroids_numba(val,clusters_nba,centroids_nba,ref)
    centroids_nba = torch.FloatTensor(centroids_nba).to(device)
    centroids[...] = centroids_nba
    # print("Any 10? ",torch.any(clusters[ref] >= 10).item())
    # print("Any < 0? ",torch.any(clusters[ref] < 0).item())
    # print(centroids.shape)
    # print(clusters.shape)

@njit
def fill_ref_centroids_numba(val,clusters,centroids,ref):
    t = clusters.shape[0]
    c,tK,s,h,w,ps,ps = centroids.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for ci in prange(c):
                    for pi in prange(ps):
                        for pj in prange(ps):
                            cid = clusters[ref,si,hi,wi].item()
                            centroids[ci,cid,si,hi,wi,pi,pj] = val
                            # try:
                            #     centroids[ci,cid,si,hi,wi,pi,pj] = val
                            # except:
                            #     print(ci,cid,si,hi,wi,pi,pj)
