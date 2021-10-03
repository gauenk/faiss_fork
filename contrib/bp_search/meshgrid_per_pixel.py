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
        div = 1
        for g in range(ngroups):
            # nunique = nuniques[hIdx,wIdx,g]
            for s in range(nsearch):
                sG = (s // div) % nsearch_per_group
                smesh[s,hIdx,wIdx,g,0] = sranges[sG,hIdx,wIdx,g,0]
                smesh[s,hIdx,wIdx,g,1] = sranges[sG,hIdx,wIdx,g,1]
            if g != refG:
                div = div * nsearch_per_group

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
    

