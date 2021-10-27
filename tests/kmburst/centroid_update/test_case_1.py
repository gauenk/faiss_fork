
# -- python --
import pytest

# -- pytorch --
import torch

# -- faiss --
import sys
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.centroid_update_utils import CENTROID_TYPE,centroid_setup

@pytest.mark.centroid
@pytest.mark.case1
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

    # -- create tensors --
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    burst,block_gt = centroid_setup(k,t,h,w,c,ps,std,device)
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    centroids = torch.zeros_like(zinits.centroids)

    # -- execute test --
    exec_test(CENTROID_TYPE,0,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,
              std,burst,block_gt,search_frames,zinits.search_ranges,
              zinits.outDists,zinits.outInds,zinits.modes,zinits.km_dists,
              centroids,zinits.clusters,zinits.cluster_sizes,zinits.blocks,zinits.ave)

    # -- compute using python --
    centroids_gt = torch.ones_like(zinits.centroids)
    
    # -- compare results --
    delta = torch.sum(torch.abs(centroids - centroids_gt)).item()
    assert delta < 1e-8, "Difference must be smaller than tolerance."


    
