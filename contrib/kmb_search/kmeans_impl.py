
import torch
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat

from pyutils.images import save_image

from .centroid_update_impl import update_centroids,update_ecentroids,fill_ecentroids
from .cluster_update_impl import update_clusters,init_clusters,rand_clusters,sup_clusters
from .pwd_impl import compute_pairwise_distance,compute_epairwise_distance

def sup_kmeans(burst,clean,indices,indices_gt,sframes,ps):
    # --> return centroids with known clustering <--
    # print("sup clusters.")
    # print("clean.shape: ",clean.shape)
    # print("burst.shape: ",clean.shape)
    # print("indices.shape: ",indices.shape)
    clusters,sizes = sup_clusters(clean,indices,indices_gt,sframes,ps)
    # print("clusters.shape: ",clusters.shape)
    # print(clusters[:,0,8,7])
    # print(clusters[:,1,8,7])
    # print(clusters[:,8,8,7])
    # print("update centroids.")
    centroids = update_ecentroids(burst,indices,clusters,sizes,ps)

    # -- aug size to modify centroids --
    c,t,s,h,w,ps,ps = centroids.shape
    aug_sizes = repeat(sizes,'t s h w -> c t s h w p1 p2',c=c,p1=ps,p2=ps)

    # -- set zero locations to nan --
    centroids[torch.where(aug_sizes==0)]=float("nan")

    # -- create ref centroid --
    inds = clusters[t//2].type(torch.long)
    inds = repeat(inds,'s h w -> c 1 s h w p1 p2',c=c,p1=ps,p2=ps)
    ref_centroid = torch.gather(centroids,dim=1,index=inds)[:,0]

    # print("return.")    
    return None,clusters,sizes,centroids,ref_centroid
    
def run_kmeans(burst,blocks,kmeansK,ps,offset,niters=10):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,nblocks,h,w = blocks.shape

    # -- init alg --
    clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)
    centroids = update_centroids(burst,blocks,clusters,sizes)

    # -- run loop --
    for iter_i in range(niters):
        
        # -- compute pwd --
        dists = compute_pairwise_distance(burst,blocks,centroids,ps,offset)

        # -- update clusters --
        clusters,sizes = update_clusters(dists)

        # -- update centroids --
        centroids = update_centroids(burst,blocks,clusters,sizes,ps)


    return dists,clusters,sizes,centroids

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

def run_ekmeans(burst,blocks,kmeansK,ps,offset,niters=10):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,nblocks,h,w = blocks.shape

    # -- init alg --
    # clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)
    clusters,sizes = rand_clusters(t,kmeansK,nblocks,h,w,device)
    # print("[kmeans.burst] max,min: ",burst.max().item(),burst.min().item())
    centroids = update_ecentroids(burst,blocks,clusters,sizes,ps)
    # print("[kmeans.centroids] max,min: ",centroids.max().item(),centroids.min().item())
    # assert torch.all(sizes == 1).item() == True,"all ones."
    # ecentroids,_,_ = fill_ecentroids(burst,blocks,ps,kmeansK)
    # delta = torch.sum(torch.abs(centroids-ecentroids)).item()
    # assert delta < 1e-8,"delta must be small here."

    # -- testing --
    tclusters,tsizes = init_clusters(t,t,nblocks,h,w,device)
    eburst = update_ecentroids(burst,blocks,tclusters,tsizes,ps)
    tdists = pwd_silly(centroids,eburst)
    
    # -- run loop --
    prevs = edict()
    prevs.centroids = centroids.clone()

    for iter_i in range(niters):
        
        # print("centroids.shape: ",centroids.shape)
        # print(eburst[0,:,0,8,7,:,:])
        # print(centroids[0,:,0,8,7,:,:])
        # print(clusters[:,0,8,7])
        # print(sizes[:,0,8,7])

        # -- compute pwd --
        # print("pre dist.")
        dists = compute_epairwise_distance(burst,blocks,centroids,offset)

        # -- tests --
        # tdists = pwd_silly(centroids,eburst).to(dists.device)
        # dists = tdists
        # tclusters = torch.argmin(dists,dim=1)
        # delta = torch.abs(tclusters - clusters).type(torch.float)
        # delta = torch.sum(delta).item()
        # print("-> Delta: ",delta)
        # print("dists.shape: ",dists.shape)
        # print(dists[:,:,0,8,7])
        # print(dists[:,:,10,8,7])
        # print("tdists.shape: ",tdists.shape)
        # delta = torch.mean(torch.abs( dists - tdists )).item()
        # print("Delta: ",delta)
        # exit()

        # -- update clusters --
        print("update clusters.")
        clusters,sizes = update_clusters(dists)
        # clusters = torch.argmin(dists,dim=1)
        # assert torch.all(sizes == 1).item() == True,"all ones."
        # delta = torch.abs(clusters - tclusters)
        # delta = torch.mean(delta.type(torch.float)).item()
        # print("Delta: ",delta)        

        # -- update centroids --
        print("update centroids.")
        prevs.centroids = centroids.clone()
        centroids = update_ecentroids(burst,blocks,clusters,sizes,ps)
        delta = torch.sum(torch.abs(centroids-prevs.centroids)).item()
        print("dCentroids: ",delta)
        # dimg = centroids!=prevs.centroids
        # print(dimg.shape)
        # save_image("kmb_dimg.png",dimg)


    return dists,clusters,sizes,centroids

def run_kmeans2weights(burst,blocks,kmeansK,ps,offset,niters=10):
    
    # -- run kmeans --
    output = run_ekmeans(burst,blocks,kmeansK,ps,offset,niters=niters)
    dists,clusters,sizes,centroids = output

    # -- create weights --
    

    
