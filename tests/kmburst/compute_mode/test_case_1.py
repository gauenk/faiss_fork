
# -- python --
import pytest
import numpy as np
from einops import rearrange,repeat

# -- pytorch --
import torch

# -- project --
from pyutils import save_image,get_img_coords

# -- faiss --
import sys
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,update_clusters,compute_pairwise_distance,init_clusters,update_centroids,run_kmeans
from kmb_search import compute_mode_pairs,compute_mode_burst,compute_mode_centroids
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.cmode_utils import CMODE_TYPE,cmode_setup

@pytest.mark.cmode
@pytest.mark.cmode_case1
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
    tol = 1e-7
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    if verbose: print(zinits.shapes)

    # -- create tensors --
    burst,offset_gt = cmode_setup(k,t,h,w,c,ps,std,device,seed)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)

    # -- tensors to test --
    modes = zinits.modes.clone()
    centroids = zinits.centroids
    dists = zinits.km_dists
    clusters = zinits.clusters
    sizes = zinits.cluster_sizes

    # -- execute test --
    exec_test(CMODE_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              zinits.outDists,zinits.outInds,modes,zinits.km_dists,
              zinits.self_dists,zinits.centroids,clusters,sizes,blocks,zinits.ave)
    modes_pair = modes.clone()
    exec_test(CMODE_TYPE,2,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              zinits.outDists,zinits.outInds,modes,zinits.km_dists,
              zinits.self_dists,zinits.centroids,clusters,sizes,blocks,zinits.ave)
    modes_burst = modes.clone()
    exec_test(CMODE_TYPE,3,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              zinits.outDists,zinits.outInds,modes,zinits.km_dists,
              zinits.self_dists,zinits.centroids,clusters,sizes,blocks,zinits.ave)
    modes_cents = modes.clone()

    # -- compute using python --
    modes_pair_gt = zinits.modes.clone()
    mode = compute_mode_pairs(std,c,ps)
    modes_pair_gt[...] = mode

    modes_burst_gt = zinits.modes.clone()
    mode = compute_mode_burst(std,c,ps,t)
    modes_burst_gt[...] = mode

    modes_cents_gt = zinits.modes.clone()
    mode = compute_mode_centroids(std,c,ps,sizes)
    modes_cents_gt[...] = mode
        
    # -- visually compare results --
    print("-- pair --")
    print(modes_pair[0,0,0,0].item())
    print(modes_pair_gt[0,0,0,0].item())

    print("-- burst --")
    print(modes_burst[0,0,0,0].item())
    print(modes_burst_gt[0,0,0,0].item())

    print("-- cents --")
    print(modes_cents[:,:,2,2])
    print(modes_cents_gt[:,:,2,2])

    # -- compare results --
    delta = torch.mean(torch.abs(modes_pair - modes_pair_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."
    delta = torch.mean(torch.abs(modes_burst - modes_burst_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."
    delta = torch.mean(torch.abs(modes_cents - modes_cents_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."

    

    
