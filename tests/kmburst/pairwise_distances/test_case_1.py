

# -- python --
import sys
import pytest
from einops import rearrange

# -- pytorch --
import torch
import torchvision
th_pad = torchvision.transforms.functional.pad

# -- project --
from pyutils import save_image,get_img_coords

# -- faiss --
import faiss
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,compute_self_pairwise_distance
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.pwd_utils import pwd_setup,PWD_TYPE

@pytest.mark.pwd_ss
@pytest.mark.case1
def test_case_1():

    # -- params --
    k = 1
    t = 4
    h = 8
    w = 8
    c = 3
    ps = 3
    nsiters = 2 # num search iters
    kmeansK = 3
    nsearch_xy = 3
    nfsearch = 3 # num of frames searched (per iter)
    nbsearch = nsearch_xy**2 # num blocks searched (per frame)
    nblocks = nbsearch**(kmeansK-1)
    std = 20./255.
    device = 'cuda:0'
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    verbose = False

    # -- create tensors --
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    burst,offset_gt = pwd_setup(k,t,h,w,c,ps,std,device)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    blocks = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2).to(device)
    dists = torch.zeros_like(zinits.self_dists)
    if verbose: print(zinits.shapes)

    # -- compute using cpp --
    exec_test(PWD_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,std,
              burst,block_gt,search_frames,zinits.search_ranges,zinits.outDists,
              zinits.outInds,zinits.modes,zinits.modes3d,zinits.km_dists,
              dists,zinits.centroids,zinits.clusters,zinits.cluster_sizes,
              blocks,zinits.ave,zinits.vals)

    # -- compute using python --
    dists_gt = compute_self_pairwise_distance(burst,blocks,ps)
    
    #
    # -- compare results --
    #

    # -- params --
    tol = 1e-4
    if verbose:
        # -- print side-by-side results --
        # midBlk = slice(nblocks//2-3,nblocks//2+3)
        midBlk = slice(nblocks-2,nblocks)
        for ti in range(t):
            print(blocks[:,ti,midBlk,4,4])
        compare = torch.stack([dists,dists_gt],dim=-1)
        print(compare[torch.where(dists_gt != 0)].cpu().numpy())
        print("-- frame 0 --")
        print(dists[0,:,midBlk,4,4])
        print(dists_gt[0,:,midBlk,4,4])
        print("-- frame 1 --")
        print(dists[1,:,midBlk,4,4])
        print(dists_gt[1,:,midBlk,4,4])
        print("-- frame 2 --")
        print(dists[2,:,midBlk,3,2])
        print(dists_gt[2,:,midBlk,3,2])
        print("-"*15)
        print("-- inspect (t0,t1) = (2,1) @ (block = -1) --")
        print("-"*15)
        print("blocks")
        print(blocks[:,2,0,:,:])
        print(blocks[:,1,0,:,:])
        print("dists")
        a_dist = dists[2,1,-1,:,:]
        b_dist = dists_gt[2,1,-1,:,:]
        print(a_dist)
        print(b_dist)
        print("delta")
        print(torch.abs(a_dist - b_dist) < tol)
        print("-"*15)
    
        # -- any elem equal? --
        elem = dists[1,2,3,4,4].item()
        elem_any = torch.any(torch.abs(dists_gt - elem)<tol).item()
        print("Any Elem?",elem,elem_any)
        
        # -- save image of equal locations --
        eqimg = torch.zeros_like(dists)
        eqimg = torch.abs(dists - dists_gt) < tol
        eqimg = eqimg.type(torch.float)
        simg = rearrange(eqimg,'t0 t1 s h w -> t0 t1 s 1 h w')
        for t0 in range(t):
            for t1 in range(t):
                timg = simg[t0,t1]
                # timg[torch.where(timg == 0)] = 0
                save_image(f"eq_img_{t0}_{t1}.png",simg[t0,t1])
    
        # -- check if output is all zero --
        print("All Zeros? ",torch.all(dists==0).item())
    
        # -- num of matches at a non-zero value --
        delta_abs = torch.abs(dists - dists_gt)
        delta_abs = delta_abs[torch.where(dists_gt != 0)]
        nmatch_nz = torch.sum(delta_abs<tol).item()
        ntotal = delta_abs.numel()
        print("Num Match @ Not Zero: %d of %d" %(nmatch_nz,ntotal))
    
    # -- test case assert --
    delta = torch.mean(torch.abs(dists - dists_gt)).item()
    assert delta < tol, "Difference must be smaller than tolerance."

