
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
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,update_clusters,compute_pairwise_distance,init_clusters
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.cluster_update_utils import CLUSTER_TYPE,cluster_setup

@pytest.mark.clu
@pytest.mark.clu_case2
def test_case_2():
    """
    Testing the clustering using the pairwise
    distance used in kmeans
    """

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
    verbose = True

    # -- create tensors --
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    burst,offset_gt = cluster_setup(k,t,h,w,c,ps,std,device,seed)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    centroids = torch.rand_like(zinits.centroids)
    dists = compute_pairwise_distance(burst,blocks,centroids,ps)
    clusters = torch.zeros_like(zinits.clusters)
    cluster_sizes = torch.zeros_like(zinits.cluster_sizes)
    if verbose: print(zinits.shapes)

    # -- execute test --
    exec_test(CLUSTER_TYPE,2,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,
              std,burst,block_gt,search_frames,zinits.search_ranges,
              zinits.outDists,zinits.outInds,zinits.modes,dists,zinits.self_dists,
              zinits.centroids,clusters,cluster_sizes,zinits.blocks,zinits.ave)
    print(clusters)

    # -- compute using python --
    # clusters_gt,csizes_gt = init_clusters(dists)
    clusters_gt,csizes_gt = update_clusters(dists)
    # print(clusters_gt)
    # print(torch.stack([clusters_gt,clusters],dim=-1))
    
    # -- num of equal --
    numEq = (clusters == clusters_gt).type(torch.float)
    print(numEq.shape)
    numEq_ave = torch.mean(numEq).item()
    print("numEq_ave: ",numEq_ave)
    numEq = rearrange(numEq,'t s h w -> t s 1 h w')
    save_image(numEq,"numEq.png")

    # -- examples neq  --
    neq = clusters != clusters_gt
    print(clusters_gt[torch.where(neq)])
    
    # -- compare results --
    delta = torch.sum(torch.abs(clusters - clusters_gt)).item()
    assert delta < 1e-8, "Difference must be smaller than tolerance."
    delta = torch.sum(torch.abs(cluster_sizes - csizes_gt)).item()
    assert delta < 1e-8, "Difference must be smaller than tolerance."


    
