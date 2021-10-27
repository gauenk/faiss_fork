# -- python imports --
import torch
import torchvision
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import swig_ptr_from_UInt8Tensor,swig_ptr_from_HalfTensor,swig_ptr_from_FloatTensor,swig_ptr_from_IntTensor,swig_ptr_from_IndicesTensor,swig_ptr_from_BoolTensor
th_pad = torchvision.transforms.functional.pad
from easydict import EasyDict as edict

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from align.xforms import align_from_pix,align_from_flow
from pyutils import get_img_coords


# ------------------------------
#
#      Manage Coordinates
#
# ------------------------------

def locs2flow(locs):
    # flows = rearrange(locs,'t i h w p two -> p i (h w) t two')
    flows_y = -locs[...,0]
    flows_x = locs[...,1]
    flows = torch.stack([flows_x,flows_y],dim=-1)
    return flows

def flow2locs(flow):
    # flow = rearrange(flow,'t i h w p two -> p i (h w) t two')
    locs_y = flow[...,0]
    locs_x = -flow[...,1]
    locs = torch.stack([locs_x,locs_y],dim=-1)
    return locs

def pix2locs(pix):
    nframes,nimages,h,w,k,two = pix.shape
    lnames = loc_index_names(1,h,w,k,pix.device)
    # -- (y,x) -> (x,y) --
    pix_y = pix[...,0]
    pix_x = pix[...,1]
    pix = torch.stack([pix_x,pix_y],dim=-1)

    # -- (x_new,y_new) -> (x_offset,y_offset) --
    locs = pix - lnames
    return locs


# ------------------------------
#
#    Warp Burst From Pix
#
# ------------------------------

def warp_burst_from_pix(burst,pix,nblocks=None):
    ndim = burst.dim()
    if ndim == 5:
        return warp_burst_from_pix_5d_burst(burst,pix,nblocks)
    if ndim == 4:
        return warp_burst_from_pix_4d_burst(burst,pix)
    else:
        msg = f"Uknown warp_burst_from_pix for burst with dims {ndim}"
        raise NotImplemented(msg)

def warp_burst_from_pix_5d_burst(burst,pix,nblocks):

    # -- block ranges per pixel --
    assert burst.dim() == 5,"The image batch dim is included"
    nframes,nimages,h,w,k,two = pix.shape
    nparticles = k
    pix = rearrange(pix,'t i h w p two -> p i (h w) t two')

    # -- create offsets --
    warps = []
    for p in range(nparticles):
        warped = align_from_pix(burst,pix[p],nblocks)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)

    return warps

def warp_burst_from_pix_4d_burst(burst,pix):

    # -- block ranges per pixel --
    two,nframes,nsearch,h,w = pix.shape
    assert two == 2,"Input shape starts with two."

    # -- pix 2 locs --
    coords = get_img_coords(nframes,nsearch,h,w)
    locs = pix - coords.to(pix.device)
    return warp_burst_from_locs_4d_burst(burst,locs,None)

def warp_burst_from_locs(burst,locs,isize=None):
    ndim = burst.dim()
    if ndim == 5:
        return warp_burst_from_locs_5d_burst(burst,locs,isize)
    elif ndim == 4:
        return warp_burst_from_locs_4d_burst(burst,locs,isize)
    else:
        msg = f"Uknown warp_burst_from_pix for burst with dims {ndim}"
        raise NotImplemented(msg)

def warp_burst_from_locs_4d_burst(burst,locs,isize):

    # -- block ranges per pixel --
    two,nframes,nblocks,h,w = locs.shape
    if isize is None: isize = edict({"h":h,"w":w})
    assert two == 2,"Input shape starts with two."

    # -- burst shape --
    burst = rearrange(burst,'c t h w -> t 1 c h w')

    # -- locs 2 flow --
    locs = rearrange(locs,'two t s h w -> t 1 h w s two')
    flows = locs2flow(locs)
    flows = rearrange(flows,'t i h w p two -> p i (h w) t two')

    # -- create offsets --
    warps = []
    for b in range(nblocks):
        warped = align_from_flow(burst,flows[b],1,isize=isize)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)

    # -- shape back --
    warps = rearrange(warps,'s t 1 c h w ->  s c t h w')
    return warps

def warp_burst_from_locs_5d_burst(burst,locs,isize):

    # -- block ranges per pixel --
    #burst.shape = t,i,c,h,w
    nframes,nimages,h,w,k,two = locs.shape
    nparticles = k
    if isize is None: isize = edict({"h":h,"w":w})
    assert two == 2,"Input shape ends with two."

    # -- locs 2 flow --
    flows = locs2flow(locs)
    flows = rearrange(flows,'t i h w p two -> p i (h w) t two')

    # -- create offsets --
    warps = []
    for p in range(nparticles):
        warped = align_from_flow(burst,flows[p],1,isize=isize)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)
    return warps
