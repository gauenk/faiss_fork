
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
from kmb_search.testing.kmbave_utils import KMBAVE_TYPE,kmbave_setup

@pytest.mark.kmbave
@pytest.mark.kmbave_case1
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
    burst,offset_gt = kmbave_setup(k,t,h,w,c,ps,std,device,seed)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)
    centroids = torch.rand_like(zinits.centroids)

    # -- tensors to test --
    ave = zinits.ave.clone()

    # -- execute test --
    exec_test(KMBAVE_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              zinits.outDists,zinits.outInds,zinits.modes,zinits.km_dists,
              zinits.self_dists,centroids,zinits.clusters,
              zinits.cluster_sizes,zinits.blocks,ave)

    # -- compute using python --
    ave_gt = torch.mean(centroids,dim=1)

    # -- visually compare --
    print(ave)
    print(ave_gt)
    print(torch.stack([ave,ave_gt],dim=-1))
        
    # -- compare results --
    delta = torch.mean(torch.abs(ave - ave_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."
    

    
