

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
from nnf_share import loc_index_names


def split_locs(pix,names,sranges,nblocks):

    # -- unpack shapes --
    nframes,nimages,h,w,k,two = pix.shape
    names = rearrange(names,'i t 1 h w -> t i h w 1')
    nframes,nimages,hN,wN,one = names.shape
    names = names[...,0]
    nsearch,nimages,h,w,nelems,two = sranges.shape
    lnames = loc_index_names(1,h,w,k,pix.device)
    nblocks2 = nblocks*nblocks
    nbHalf = nblocks//2
    gsize = (nblocks + 2 * nbHalf)**2

    # -- (y,x) -> (x,y) --
    pix_y = pix[...,0]
    pix_x = pix[...,1]
    pix = torch.stack([pix_x,pix_y],dim=-1)

    # -- (x_new,y_new) -> (x_offset,y_offset) --
    locs = pix - lnames
    locs = locs[...,0,:]
    assert h == hN, f"[names and locs] two sizes must match: {h} vs {hN}"
    assert w == wN, f"[names and locs] two sizes must match: {w} vs {wN}"

    # -- cuda gpuid --
    gpuid = pix.device.index

    hIdx,wIdx = 16,15
    # print(sranges[:,:,hIdx,wIdx,:,0])
    # print(sranges[nsearch//2,:,hIdx,wIdx,:,0])
    # print(sranges[nsearch//2,:,hIdx,wIdx,:,1])
    # print(locs[:,0,16,16,:])

    # -- run launcher --
    glocs = []
    split_search_ranges = []
    ngroups = names.max()+1
    for i in range(nimages):
        glocs_i = torch.zeros(gsize,h,w,ngroups,2).to(gpuid).type(torch.long)
        for groupID in range(ngroups):
            split_locs_launcher(glocs_i,locs[:,i],names[:,i],
                                sranges[:,i],nblocks,groupID,gpuid)
            # print(glocs_i[0,0,0,0,groupID,:])
            print("-- glocs[...,groupID,..] --")
            print(glocs_i[:,16,16,groupID,:])
            print("-"*10)
            # print(glocs_i[...,groupID,:].shape)
        # print(glocs_i[:,16,16,:,:])
        # print("glocs_i.shape ",glocs_i[:,16,16,:,:].shape)
        # print("sranges: ",sranges[:,0].shape)
        print("glocs_i.shape ",glocs_i.shape)
        print("sranges: ",sranges[:,0].shape)

        split_search_ranges_i = glocs_i 
        split_search_ranges.append(split_search_ranges_i)
        glocs.append(glocs_i)
    glocs = torch.stack(glocs)
    ssr = torch.stack(split_search_ranges) # nimages,nsearch,h,w,ngroups,two
    # print(ssr)
    print(ssr.shape)
    print(locs.shape)
    # for i in range(h):
    #     for j in range(w):
    #         if i == 16 and j == 16:
    #             print("--")
    #             print(i,j)
    #             print("-- ssr --")
    #             print(ssr[:,:,i,j])
    #             print("-- locs --")
    #             print(locs[:,0,i,j])

    return glocs 


@cuda.jit
def split_locs_numba_cuda(glocs,offsets,locs,names,sranges,nblocks,grid,groupID):

    # -- get kernel indices --
    blockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
    threadId = blockId * ( cuda.blockDim.x * cuda.blockDim.y ) + ( cuda.threadIdx.x * cuda.blockDim.x ) + cuda.threadIdx.x
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

            x = locs[t,hIdx,wIdx,0]
            y = locs[t,hIdx,wIdx,1]

            best_x = locs[t,hIdx,wIdx,0]
            best_y = locs[t,hIdx,wIdx,1]
            name = names[t,hIdx,wIdx]

            x_offset = int(x - sranges[nsearch//2,hIdx,wIdx,t,0])
            y_offset = int(y - sranges[nsearch//2,hIdx,wIdx,t,1])
            # assert np.abs(x_offset) <= nbHalf,"centered must be contained."
            # assert np.abs(y_offset) <= nbHalf,"centered must be contained."
            
            # offsets -> block index
            x_bindex = x_offset + nbHalf
            y_bindex = -y_offset + nbHalf # spatial to matrix coords
            # index of "best" position within (nblock,nblock) matrix
            # x == column, y == row

            x_tl = mid - x_bindex
            y_tl = mid - y_bindex

            x_offset = best_x + nbHalf
            y_offset = best_y + nbHalf

            bl_x = x_offset - x_tl
            bl_y = y_offset - y_tl

            # off_x = bl_x - nbHalf
            # off_y = -bl_y + nbHalf
            # l_x = x_tl - tmp_x
            # l_y = y_tl - tmp_y

            off_x = bl_x - nbHalf
            off_y = -(bl_y - nbHalf)

            if name == groupID: val = 1
            else: val = 0
            nelems += val

            # if val == 1:
            #     grid[hIdx][wIdx][0][0] = x_tl
            #     grid[hIdx][wIdx][0][1] = y_tl


            # xslice = slice(x_tl,x_tl+nblocks)
            # yslice = slice(y_tl,y_tl+nblocks)

            for y_b in range(nblocks):
                for x_b in range(nblocks):
                    y_i = y_tl + y_b
                    x_i = x_tl + x_b
                    # grid[hIdx][wIdx][0][0] = hIdx
                    # grid[hIdx][wIdx][0][1] = wIdx
                    yvalid = True#y_i < grid.shape[2] and 0 <= y_i
                    xvalid = True#x_i < grid.shape[3] and 0 <= x_i
                    if val == 1 and yvalid and xvalid:
                        grid[hIdx][wIdx][y_i][x_i] += 1

        # for y_b in range(nbHalf,nbHalf+nblocks):
        #     for x_b in range(nbHalf,nbHalf+nblocks):
        offset_x,offset_y = 0,0
        first = True
        for y_b in range(0,grid.shape[2]):
            for x_b in range(0,grid.shape[3]):
                if grid[hIdx,wIdx,y_b,x_b] == nelems:
                    glocs[sIdx,hIdx,wIdx,groupID,0] = x_b# - mid
                    glocs[sIdx,hIdx,wIdx,groupID,1] = y_b# - mid
                    if first:
                        offset_x = x_b + nbHalf
                        offset_y = y_b + nbHalf
                        first = False
                else:
                    glocs[sIdx,hIdx,wIdx,groupID,0] = SILLY_BIG # no motion
                    glocs[sIdx,hIdx,wIdx,groupID,1] = SILLY_BIG # no motion
                # ycond = nbHalf < y_b and y_b < (nblocks + nbHalf)
                # xcond = nbHalf < x_b and x_b < (nblocks + nbHalf)
                # if ycond and xcond:
                scond = sIdx < glocs.shape[0]
                sIdx += 1


def split_locs_launcher(glocs,locs,names,sranges,nblocks,groupID,gpuid):

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
    grid = torch.zeros(gsize,gsize).to(gpuid).type(torch.long)
    grid = repeat(grid,'g1 g2 -> h w g1 g2',h=height,w=width).clone()
    # print("grid.shape ",grid.shape)

    # -- convert elems to numba cuda --
    glocs_nb = numba.cuda.as_cuda_array(glocs)
    locs = numba.cuda.as_cuda_array(locs)
    names = numba.cuda.as_cuda_array(names)
    sranges = numba.cuda.as_cuda_array(sranges)
    grid_nb = numba.cuda.as_cuda_array(grid)

    # -- launch config --
    threads_per_block = (8,8)
    blocks_h = height//threads_per_block[0] + (height%threads_per_block[0] != 0)
    blocks_w = width//threads_per_block[1] + (width%threads_per_block[1] != 0)
    blocks = (blocks_h,blocks_w)
    print(blocks)

    # -- launch! --
    split_locs_numba_cuda[blocks,threads_per_block](glocs_nb,locs,names,
                                                    sranges,nblocks,grid_nb,groupID)

    print("--- grids ---")
    print(grid[16,15])
    print(grid[15,16])
    print(grid[16,16])
