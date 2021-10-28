
# -- python --
import pytest
from einops import rearrange,repeat

# -- pytorch --
import torch

# -- project --
from pyutils import save_image,get_img_coords

# -- faiss --
import sys
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,update_clusters,compute_pairwise_distance,init_clusters,update_centroids
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.centroid_update_utils import CENTROID_TYPE,centroid_setup

@pytest.mark.centu
@pytest.mark.centu_case1
def test_case_1():

    # -- params --
    k = 1
    t = 4
    h = 8
    w = 8
    c = 3
    ps = 11
    nsiters = 2 # num search iters
    kmeansK = 3
    nsearch_xy = 3
    nfsearch = 3 # num of frames searched (per iter)
    nbsearch = nsearch_xy**2 # num blocks searched (per frame)
    nblocks = nbsearch**(kmeansK-1)
    std = 20./255.
    device = 'cuda:0'
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    seed = 234
    verbose = False
    tol = 1e-6

    # -- create tensors --
    burst,offset_gt = centroid_setup(k,t,h,w,c,ps,std,device,seed)
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    centroids = torch.rand_like(zinits.centroids)
    dists = compute_pairwise_distance(burst,blocks,centroids,ps)
    clusters,cluster_sizes = update_clusters(dists)
    # cluster_sizes = torch.zeros_like(zinits.cluster_sizes)
    if verbose: print(zinits.shapes)

    # -- execute test --
    exec_test(CENTROID_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              zinits.outDists,zinits.outInds,zinits.modes,dists,
              zinits.self_dists,centroids,clusters,cluster_sizes,
              blocks,zinits.ave)

    # -- compute using python --
    centroids_gt = update_centroids(burst,blocks,clusters,cluster_sizes)
    if verbose:
        print(torch.all(centroids_gt == 0).item())
        print(centroids_gt.shape)
        print(torch.stack([centroids[:,:,0,0,0],centroids_gt[:,:,0,0,0]],dim=-1))
        print(torch.stack([centroids[:,:,21,3,3],centroids_gt[:,:,21,3,3]],dim=-1))
    
        adiff = torch.abs(centroids - centroids_gt)
        perc_eq = torch.mean((adiff < tol).type(torch.float)).item()
        print("Percent Equal: %2.3f" % (perc_eq))
        print(adiff.shape)
        adiff = rearrange(adiff,'c t s h w -> c t s 1 h w').type(torch.float)
        save_image("neq.png",adiff)
        
    # -- compare results --
    delta = torch.mean(torch.abs(centroids - centroids_gt)).item()
    assert delta < 1e-8, "Difference must be smaller than tolerance."


    
