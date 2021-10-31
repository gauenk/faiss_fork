
# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

def tiled_search_frames(nfsearch,nsiters,ref):
    sframes = torch.zeros(nsiters,nfsearch)
    base = torch.arange(nfsearch).type(torch.int)
    for i in range(nsiters):
        sframes[i,:] = base + i
        if len(torch.where(ref == sframes[i,:])[0]) == 0:
               sframes[i,0] = ref
    sframes = sframes.type(torch.int)
    return sframes

def mesh_from_ranges(search_ranges,search_frames,curr_blocks,ref):
    # -- create mesh of current blocks --
    search_frames = search_frames.type(torch.long) # for torch indexing
    two,t,s,h,w = search_ranges.shape
    sranges = rearrange(search_ranges,'two t s h w -> s 1 h w t two')
    sranges = sranges[...,search_frames,:]
    ref = torch.where(search_frames == ref)[0][0].item()
    mesh = create_mesh_from_ranges(sranges,ref)
    mesh = rearrange(mesh,'b 1 h w g two -> two g b h w')

    # -- append the current state frames --
    nblocks = mesh.shape[-3]
    blocks = curr_blocks.clone()
    blocks = repeat(blocks,'two t h w -> two t b h w',b=nblocks)
    for group,frame in enumerate(search_frames):
        if frame == ref: continue
        blocks[:,frame] = mesh[:,group]
    
    return blocks

def jitter_search_ranges(nrange,t,h,w):

    # -- create ranges --
    mrange = nrange//2
    sranges = torch.zeros(nrange,nrange,2)
    for i in range(nrange):
        for j in range(nrange):
            sranges[i,j,0] = i - mrange
            sranges[i,j,1] = j - mrange
    sranges = rearrange(sranges,'r1 r2 two -> (r1 r2) two')
    
    # -- repeat to full shape --
    sranges = repeat(sranges,'r2 two -> two t r2 h w',t=t,h=h,w=w)
    sranges = sranges.type(torch.int).contiguous()

    # -- compute absolute coordinates --
    r2 = nrange**2
    coords = get_img_coords(t,r2,h,w)

    # -- from relative to absolute coordinates --
    sranges = sranges + coords

    return sranges

def jitter_traj_ranges(trajs,jsize):
    k,i,two,t,h,w = trajs.shape
    jitter = jitter_search_ranges(jsize,t,h,w).to(trajs.device)
    jitter = repeat(jitter,'two t r2 h w -> k i two t r2 h w',k=k,i=i)
    trajs = repeat(trajs,'k i two t h w -> k i two t r2 h w',r2=jsize**2)
    jtrajs = trajs + jitter
    return jtrajs

def init_zero_traj(nframes,nimages,h,w):
    locs = init_zero_locs(nframes,nimages,h,w)
    trajs = rearrange(locs,'t i h w k two -> k i two t h w')        
    return trajs

def init_zero_locs(nframes,nimages,h,w):
    locs = torch.zeros(nframes,nimages,h,w,1,2)
    locs = locs.type(torch.int)
    return locs

def compute_l2_mode(std,patchsize):
    return 0.


