
import torch

from .centroid_update_impl import update_centroids,update_ecentroids,fill_ecentroids
from .cluster_update_impl import update_clusters,init_clusters
from .pwd_impl import compute_pairwise_distance,compute_epairwise_distance

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

def run_ekmeans(burst,blocks,kmeansK,ps,offset,niters=10):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    two,t,nblocks,h,w = blocks.shape

    # -- init alg --
    clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)
    # print("[kmeans.burst] max,min: ",burst.max().item(),burst.min().item())
    centroids = update_ecentroids(burst,blocks,clusters,sizes,ps)
    # print("[kmeans.centroids] max,min: ",centroids.max().item(),centroids.min().item())
    # assert torch.all(sizes == 1).item() == True,"all ones."
    # ecentroids,_,_ = fill_ecentroids(burst,blocks,ps,kmeansK)
    # delta = torch.sum(torch.abs(centroids-ecentroids)).item()
    # assert delta < 1e-8,"delta must be small here."

    # -- run loop --
    for iter_i in range(niters):
        
        # -- compute pwd --
        dists = compute_epairwise_distance(burst,blocks,centroids,offset)
        
        # -- update clusters --
        clusters,sizes = update_clusters(dists)
        # assert torch.all(sizes == 1).item() == True,"all ones."

        # -- update centroids --
        centroids = update_ecentroids(burst,blocks,clusters,sizes,ps)

    return dists,clusters,sizes,centroids

