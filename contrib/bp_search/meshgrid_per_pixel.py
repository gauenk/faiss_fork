# -- python imports --
import math,operator
import torch
import numpy as np
import numba
from numba import cuda
from numba.typed import List
from einops import rearrange,repeat
from numba import njit,jit,prange


@cuda.jit
def meshgrid_per_pixel_cuda(sranges,smesh,refG):

    hIdx,wIdx = cuda.grid(2)
    nsearch_per_group,height,width,ngroups,two = sranges.shape
    nsearch,height,width,ngroups,two = smesh.shape
    
    if hIdx < height and wIdx < width:
        for s in range(nsearch):
            # nunique = nuniques[hIdx,wIdx,g]
            gmod = 0
            div = 1
            for g in range(ngroups):
                if g != refG:
                    div = nsearch_per_group**gmod
                    sG = (s // div) % nsearch_per_group
                    gmod += 1
                else:
                    sG = 0#earch_per_group//2
                smesh[s,hIdx,wIdx,g,0] = sranges[sG,hIdx,wIdx,g,0]
                smesh[s,hIdx,wIdx,g,1] = sranges[sG,hIdx,wIdx,g,1]
            # if g != refG:
            #     div = div * nsearch_per_group

def meshgrid_per_pixel_launcher(sranges,smesh,refG):

    # -- shapes --
    nsearch_per_group,height,width,ngroups,two = sranges.shape

    # -- numba-fy tensors --
    sranges_nb = numba.cuda.as_cuda_array(sranges)
    smesh_nb = numba.cuda.as_cuda_array(smesh)

    # -- kernel launch config --
    threads = (8,8)
    blocks_h = height//threads[0] + (height%threads[0] != 0)
    blocks_w = width//threads[1] + (width%threads[1] != 0)
    blocks = (blocks_h,blocks_w)

    # -- exec kernel --
    meshgrid_per_pixel_cuda[blocks,threads](sranges_nb,smesh_nb,refG)

    # print(smesh[:,16,16,0])
    # print(smesh[:,16,16,2])
    # print(smesh[:,16,16])
    # print(smesh.shape)
    
def create_mesh_from_ranges(sranges,refG,img_shape=None):

    # -- init shapes and smesh -- 
    nsearch_per_group,nimages,h,w,ngroups,two = sranges.shape
    nsearch = nsearch_per_group**(ngroups-1) # since ref only has 1 elem
    smesh = torch.zeros(nsearch,nimages,h,w,ngroups,two).type(torch.int)
    smesh = smesh.to(sranges.device)

    # -- numba-fy values --
    for i in range(nimages):
        meshgrid_per_pixel_launcher(sranges[:,i],smesh[:,i],refG)
        # smesh[:,i] = torch.flip(smesh[:,i],dims=(-1,))
    return smesh

