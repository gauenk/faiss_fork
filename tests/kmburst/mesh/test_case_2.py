
# -- python --
import sys
import pytest
from einops import rearrange

# -- pytorch --
import torch

# -- project --
from pyutils import save_image,get_img_coords

# -- faiss --
import sys
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.mesh_utils import MESH_TYPE,mesh_setup

@pytest.mark.mesh_case2
def test_case_2():

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
    verbose = False
    seed = 123

    # -- create tensors --
    zinits = init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,
                               nfsearch,kmeansK,nsiters,device)
    burst,offset_gt = mesh_setup(k,t,h,w,c,ps,std,device,seed)
    block_gt = offset_gt + coords
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nfsearch,nsiters,t//2).to(device)
    if verbose:
        print(search_frames)
        print(search_frames.shape)
        print(search_ranges.shape)
    
    # -- for testing --
    blocks = zinits.blocks
    block_eqs = (search_ranges == block_gt[:,:,None])
    block_eqs = torch.all(block_eqs,dim=0)
    block_eqs = block_eqs.type(torch.float)
    init_blocks = torch.argmax(block_eqs,dim=1,keepdim=True)[:,0]
    init_blocks = init_blocks.type(torch.int)
    if verbose:
        print("init_blocks.shape: ",init_blocks.shape)
        print(init_blocks[-1,:,:])
    
    
    # -- execute test --
    exec_test(MESH_TYPE,1,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,std,
              burst,init_blocks,search_frames,search_ranges,zinits.outDists,
              zinits.outInds,zinits.modes,zinits.modes3d,zinits.km_dists,
              zinits.self_dists,zinits.centroids,zinits.clusters,
              zinits.cluster_sizes,blocks,zinits.ave,zinits.vals)
    
    # -- compute using python --
    blocks_gt = mesh_from_ranges(search_ranges,search_frames[0],block_gt,t//2)
    blocks_gt = blocks_gt.to(device)

    # -- visually compare blocks --
    if verbose:
        print("-- blocks --")
        print(blocks.shape)
    
    
        for s in range(nblocks):
            numEq = blocks[:,:,s,:,:] == blocks_gt[:,:,s,:,:]
            numEq = numEq.type(torch.float)
            print(s,numEq.mean().item())
    
        print("-"*30)
        print(search_ranges.shape)
        for i in range(search_ranges.shape[1]):
            print(i)
            print(search_ranges[:,i,:,4,4])
        print("-"*30)
        print(blocks[:,:,0,4,4])
        print(blocks_gt[:,:,0,4,4])
        print("-"*30)
        print(blocks[:,:,7,4,4])
        print(blocks_gt[:,:,7,4,4])
        print("-"*30)
        print(blocks[:,:,40,4,4])
        print(blocks_gt[:,:,40,4,4])
        print("-"*30)
        # print(blocks[:,:,0,:,:] == blocks_gt[:,:,0,:,:])
        # print(blocks[:,:,40,:,:] == blocks_gt[:,:,40,:,:])
        # print(block_gt[:,:,0,0])
        # print(search_ranges[:,0,:,0,0])
        
    # -- compare results --
    delta = torch.sum(torch.abs(blocks - blocks_gt)).item()
    assert delta < 1e-8, "Difference must be smaller than tolerance."


    
