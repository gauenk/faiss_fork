
# -- python imports --
import math,operator
import torch
import numpy as np
import numba
from numba import cuda
from numba.typed import List
from einops import rearrange,repeat
from numba import njit,jit,prange
from pyutils import torch_to_numpy
from nnf_share import loc_index_names,locs2flow

from .utils import pix2locs

def merge_search_ranges(pix,names,sranges,nblocks,pixAreLocs=False,drift=False):

    # -- unpack shapes --
    nsearch,nimages,h,w,nframes,two = sranges.shape
    nframes,nimages,h,w,k,two = pix.shape
    names = rearrange(names,'i t 1 h w -> t i h w 1')
    nframes,nimages,hN,wN,one = names.shape
    names = names[...,0]
    nblocks2 = nblocks*nblocks
    nbHalf = nblocks//2
    gsize = (nblocks + 2 * nbHalf)**2

    # -- convert pix to locs --
    if not(pixAreLocs):
        locs = pix2locs(pix)
    else:
        locs = pix

    # -- misc --
    locs = locs[...,0,:]
    assert h == hN, f"[names and locs] two sizes must match: {h} vs {hN}"
    assert w == wN, f"[names and locs] two sizes must match: {w} vs {wN}"

    
    # -- cuda gpuid --
    gpuid = pix.device.index
    refT = nframes//2

    hIdx,wIdx = 16,15
    # print(sranges[:,:,hIdx,wIdx,:,0])
    # print(sranges[nsearch//2,:,hIdx,wIdx,:,0])
    # print(sranges[nsearch//2,:,hIdx,wIdx,:,1])
    # print(locs[:,0,16,16,:])

    # -- run launcher --
    glocs = []
    merged_search_ranges = []
    ngroups = int(names.max().item())+1
    offsets = torch.zeros(nimages,1,h,w,ngroups,2).to(gpuid).type(torch.long)
    for i in range(nimages):
        glocs_i = torch.zeros(nblocks2,h,w,ngroups,2).to(gpuid).type(torch.long)
        offsets_i = offsets[i]
        for groupID in range(ngroups):
            offsets_ig = offsets_i[0,:,:,groupID]
            merge_search_ranges_launcher(glocs_i,offsets_ig,locs[:,i],names[:,i],
                                         sranges[:,i],nblocks,groupID,refT,drift,gpuid)
            # print(glocs_i[0,0,0,0,groupID,:])
            # print("-- glocs[...,groupID,..] --")
            # print(glocs_i[:,16,16,groupID,:])
            # print("-- glocs[...,groupID,..] - offsets --")
            # print(glocs_i[:,16,16,groupID,:] - offsets_ig[16,16])
            # print("-"*10)
            # print(glocs_i[...,groupID,:].shape)
        # print(glocs_i[:,16,16,:,:])
        # print("glocs_i.shape ",glocs_i[:,16,16,:,:].shape)
        # print("sranges: ",sranges[:,0].shape)
        # print("[pre] offsets_i.shape ",offsets_i.shape)
        # offsets_i = repeat(offsets_i,'h w g two -> 1 h w g two')
        # print("glocs_i.shape ",glocs_i.shape)
        # print("offsets_i.shape ",offsets_i.shape)
        # print("sranges: ",sranges[:,0].shape)

        merged_search_ranges_i = glocs_i# - offsets_i
        merged_search_ranges.append(merged_search_ranges_i)
        glocs.append(glocs_i)
    glocs = torch.stack(glocs)
    msr = torch.stack(merged_search_ranges,dim=1) # nimages,nsearch,h,w,ngroups,two
    # print(msr)
    # print(msr.shape)
    # print(locs.shape)
    # for i in range(h):
    #     for j in range(w):
    #         if i == 16 and j == 16:
    #             print("--")
    #             print(i,j)
    #             print("-- msr --")
    #             print(msr[:,:,i,j])
    #             print("-- search ranges --")
    #             print(sranges[:,:,i,j])
    #             print("-- locs --")
    #             print(locs[:,0,i,j])
    #             print("-- offsets --")
    #             print(offsets[:,0,i,j])

    return msr,offsets


@cuda.jit
def merge_search_ranges_numba_cuda(glocs,offsets,locs,names,sranges,nblocks,grid,groupID,refT,drift):

    # -- get kernel indices --
    # blockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
    # threadId = blockId * ( cuda.blockDim.x * cuda.blockDim.y ) + ( cuda.threadIdx.x * cuda.blockDim.x ) + cuda.threadIdx.x
    hIdx,wIdx = cuda.grid(2)

    nframes,height,width,two = locs.shape
    # nframes,hN,wN = names.shape
    # nsearch,h,w,ngroups,two = glocs.shape
    nsearch,h,w,nframes,two = sranges.shape
    sIdx = 0
    nelems = 0
    nbHalf = nblocks//2
    mid = grid.shape[-1]//2

    if hIdx < height and wIdx < width:

        for t in range(nframes):
            y = locs[t,hIdx,wIdx,0]
            x = locs[t,hIdx,wIdx,1]
            name = names[t,hIdx,wIdx]

            # locs to offests: just (0,0) right now.
            y_offset = int(y - sranges[nsearch//2,hIdx,wIdx,t,0])
            x_offset = int(x - sranges[nsearch//2,hIdx,wIdx,t,1])

            # offsets -> block index
            y_bindex = y_offset + nbHalf # spatial to matrix coords
            x_bindex = x_offset + nbHalf

            y_tl = mid - y_bindex
            x_tl = mid - x_bindex

            if name == groupID: val = 1
            else: val = 0
            nelems += val

            for y_b in range(nblocks):
                for x_b in range(nblocks):
                    y_i = y_tl + y_b
                    x_i = x_tl + x_b
                    if val == 1:
                        grid[hIdx][wIdx][y_i][x_i] += 1

        # -- use center and only center if ref frame --
        if names[refT,hIdx,wIdx] == groupID:
            for y_b in range(0,grid.shape[2]):
                for x_b in range(0,grid.shape[3]):
                    grid[hIdx][wIdx][y_b][x_b] = 0
            for sIdx in range(0,glocs.shape[0]):
                glocs[sIdx,hIdx,wIdx,groupID,0] = 0
                glocs[sIdx,hIdx,wIdx,groupID,1] = 0
        else:

            if drift:
                # -- write the shared search space into output --
                first = True
                sIdx = 0
                for y_b in range(0,grid.shape[2]):
                    for x_b in range(0,grid.shape[3]):
                        if grid[hIdx,wIdx,y_b,x_b] == nelems:
                            glocs[sIdx,hIdx,wIdx,groupID,0] = y_b - mid
                            glocs[sIdx,hIdx,wIdx,groupID,1] = x_b - mid
                            if first:
                                offsets[hIdx,wIdx,0] = y_b# + nbHalf
                                offsets[hIdx,wIdx,1] = x_b# + nbHalf
                                first = False
                            sIdx += 1
        
                # -- shift the legal locs to the correctly shifted value elems --
                # sIdx = 0
                # shift_x = offsets[hIdx,wIdx,0]
                # shift_y = offsets[hIdx,wIdx,1]
                # for sIdx in range(0,glocs.shape[0]):
                #     glocs[sIdx,hIdx,wIdx,groupID,0] -= shift_x
                #     glocs[sIdx,hIdx,wIdx,groupID,1] -= shift_y
                            
            else:
                sIdx = 0
                for y_b in range(mid-nbHalf,mid+nbHalf+1):
                    for x_b in range(mid-nbHalf,mid+nbHalf+1):
                        if grid[hIdx,wIdx,y_b,x_b] == nelems:
                            glocs[sIdx,hIdx,wIdx,groupID,0] = x_b - mid
                            # glocs[sIdx,hIdx,wIdx,groupID,1] = y_b - mid # no flip?
                            glocs[sIdx,hIdx,wIdx,groupID,1] = -(y_b - mid)#flip-up-down?
                        if y_b == mid and x_b == mid:
                            offsets[hIdx,wIdx,0] = x_b - mid
                            offsets[hIdx,wIdx,1] = -(y_b - mid)
                        sIdx += 1
                    

def merge_search_ranges_launcher(glocs,offsets,locs,names,sranges,nblocks,groupID,refT,drift,gpuid):

    # -- select id --
    numba.cuda.select_device(gpuid)

    # -- get shapes --
    nframes,height,width,two = locs.shape
    # print("locs.type: ",locs.type())
    # print(locs[:,16,15,:])
    # print(locs[:,15,16,:])
    # print(locs[:,16,16,:])

    # -- create grid memory --
    gsize = nblocks + 2*(nblocks//2)
    mid = gsize//2
    grid = torch.zeros(gsize,gsize).to(gpuid).type(torch.long)
    grid = repeat(grid,'g1 g2 -> h w g1 g2',h=height,w=width).clone()
    # print("grid.shape ",grid.shape)

    # -- convert elems to numba cuda --
    glocs_nb = numba.cuda.as_cuda_array(glocs)
    # flow = locs2flow(locs)
    locs_nb = numba.cuda.as_cuda_array(locs)
    names_nb = numba.cuda.as_cuda_array(names)
    sranges = numba.cuda.as_cuda_array(sranges)
    grid_nb = numba.cuda.as_cuda_array(grid)
    offsets_nb = numba.cuda.as_cuda_array(offsets)

    # -- launch config --
    threads_per_block = (8,8)
    blocks_h = height//threads_per_block[0] + (height%threads_per_block[0] != 0)
    blocks_w = width//threads_per_block[1] + (width%threads_per_block[1] != 0)
    blocks = (blocks_h,blocks_w)

    # -- launch! --
    merge_search_ranges_numba_cuda[blocks,threads_per_block](glocs_nb,offsets_nb,
                                                             locs_nb,names_nb,
                                                             sranges,nblocks,grid_nb,
                                                             groupID,refT,drift)

    # print(f"--- GroupID [{groupID}] ---")

    # nbHalf = nblocks//2
    # hIdx = 16+nbHalf
    # wIdx = 16+nbHalf

    # print("--- locs ---")
    # print(locs.shape)
    # print(locs[:,hIdx,wIdx])

    # print("--- names ---")
    # print(names[:,hIdx,wIdx])

    # print("--- grids ---")
    # print(grid[hIdx,wIdx])

    # print("--- glocs ---")
    # print(glocs[:,hIdx,wIdx,groupID,:])

    # print("--- offsets ---")
    # print(offsets[hIdx,wIdx])
    # print(offsets[hIdx,wIdx])
    # print(offsets[hIdx,wIdx])

    # print("--- locs ---")
    # print(locs.shape)
    # print(locs[:,16+2,15+2])
    # print(locs[:,15+2,16+2])
    # print(locs[:,16+2,16+2])

    # print(grid[16+2,15+2])
    # print(grid[15+2,16+2])
    # print(grid[16+2,16+2])

    # print(grid[16,15])
    # print(grid[15,16])
    # print(grid[16,16])


