
"""
DBSCAN over the batch across time

"""

import torch
import numpy as np
from einops import rearrange,repeat
from sklearn import cluster as sk_cluster

from .cluster_update_impl import init_clusters
from .centroid_update_impl import update_ecentroids

def pwd_silly(a,b):
    c,ta,s,h,w,ps,ps = a.shape
    ta,tb = a.shape[1],b.shape[1]
    dists = torch.zeros((tb,ta,s,h,w)).to(a.device)
    for ti in range(ta):
        for tj in range(tb):
            sq_delta = (a[:,ti]-b[:,tj])**2
            sq_delta = torch.sum(sq_delta,axis=(0,4,5))
            dists[tj,ti] = sq_delta
    return dists


def run_dbscan_scikit(burst,indices,K,ps,km_offset,eps,minPts):
    
    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = indices.shape
    cluster_ps = 11

    # -- create centroids for each frame --
    clusters,sizes = init_clusters(t,t,s,h,w,device)
    eburst = update_ecentroids(burst,indices,clusters,sizes,cluster_ps)
    eburst = eburst.cpu().numpy()
    sizes = torch.zeros_like(sizes)

    # -- run dbscan for each prop --
    for si in range(s):
        for hi in range(h):
            for wi in range(w):

                # -- extract sample --
                pburst = eburst[:,:,si,hi,wi,:,:]
                pburst = rearrange(pburst,'c t p1 p2 -> t (c p1 p2)')

                # -- run dbscan --
                sk_dbscan = sk_cluster.DBSCAN(eps=eps,min_samples=minPts)
                # sk_dbscan = sk_cluster.OPTICS(min_samples=minPts)
                sk_dbscan = sk_dbscan.fit(pburst)


                # -- format output --
                cids = sk_dbscan.labels_
                cids = torch.IntTensor(cids).to(device)
                # print("c",cids)

                # -- remove -1 indices --
                max_cid = cids.max().item()+1
                for c,cid in enumerate(cids):
                    if cid == -1:
                        cids[c] = max_cid
                        max_cid += 1
                # print("b",cids)
                        
                # -- update --
                clusters[:,si,hi,wi] = cids
                sizes[cids.type(torch.long),si,hi,wi] += 1
                # print(sizes[:,si,hi,wi])

    # -- to device --
    centroids = update_ecentroids(burst,indices,clusters,sizes,ps)
    # centroids = torch.FloatTensor(centroids).to(device)

    return None,clusters,sizes,centroids

def run_dbscan(burst,indices,K,ps,km_offset,eps,minPts):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,s,h,w = blocks.shape
    clusters = torch.ByteTensor(torch.zeros((t,s,h,w)))
    nframes = t

    # -- dbscan params --
    eps = 1e-1
    minPts = 2
    
    # -- start dbscan --
    nclusters = 0
    for p in range(nframes):
        
        # -- get neighbors --
        neighbors = query_neighbors(burst,indices,p,eps)
        
        # -- filter min points --
        mask = filter_min_pts(neighbors,p,clusters,sizes,minPts)

        # -- update clusters --
        nclusters += 1

        # -- label initial --
        clusters,sizes = update_label(nclusters,p,clusters,sizes)

        for q in neighbors:
            if q == p: continue
            
            # -- update noise labels --
            clusters,sizes = update_noise_label(nclusters,q,clusters,sizes)
            
            # -- mask out already labeled points --
            mask = rm_labeled_points(q,clusters,sizes)

            # -- label non-masked points as q --
            clusters,sizes = update_label(nclusters,q,clusters,sizes,mask)

            # -- compute neighbors for undef points --
            q_neighbors = query_neighbors(burst,indices,q,eps)

            # -- filter min points --
            clusters,sizes = filter_min_pts(q_neighbors,q,clusters,sizes,minPts,mask)

            # -- update set --

    return dists,clusters,sizes,centroids

    
