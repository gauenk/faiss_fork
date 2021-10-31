
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
from kmb_search import compute_mode_pairs,compute_mode_burst,compute_mode_centroids,kmb_topk
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.kmbtopk_utils import KMBTOPK_TYPE,kmbtopk_setup

@pytest.mark.kmbtopk
@pytest.mark.kmbtopk_case1
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
    burst,offset_gt = kmbtopk_setup(k,t,h,w,c,ps,std,device,seed)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    clusters,sizes = init_clusters(t,kmeansK,nblocks,h,w,device)
    centroids = torch.rand_like(zinits.centroids)

    # -- tensors to test --
    modes3d = zinits.modes3d.clone()
    inDists = torch.rand_like(zinits.vals)
    outDists = zinits.outDists.clone()
    outInds = zinits.outInds.clone()
    ave = zinits.ave.clone()
    outDists[...] = 1000.

    # -- execute test --
    exec_test(KMBTOPK_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,block_gt,search_frames,search_ranges,
              outDists,outInds,zinits.modes,modes3d,zinits.km_dists,
              zinits.self_dists,centroids,zinits.clusters,
              zinits.cluster_sizes,blocks,zinits.ave,inDists)

    # -- compute using python --
    outDists_gt,outInds_gt = kmb_topk(inDists,blocks,k,modes3d)

    # -- visually compare --
    if verbose:
        print("outDists.shape: ",outDists.shape)
        print("outDists_gt.shape: ",outDists_gt.shape)
        print(outDists[:,0,0])
        print(outDists_gt[:,0,0])
            
    # -- compare results --
    delta = torch.mean(torch.abs(outDists - outDists_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."
    delta = torch.mean(torch.abs(outInds - outInds_gt).type(torch.float)).item()
    assert delta < tol, "Difference must be smaller than tolerance."
    

    
