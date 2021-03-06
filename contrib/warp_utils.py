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

def loc_index_names(nimages,h,w,k,device):
    npix = h*w
    locs_ref = np.c_[np.unravel_index(np.arange(npix),(h,w))]
    locs_ref = locs_ref.reshape(h,w,2)
    locs_ref = repeat(locs_ref,'h w two -> i h w k two',i=nimages,k=k)
    locs_ref = torch.IntTensor(locs_ref).to(device,non_blocking=True)
    return locs_ref

def flow2pix(flow):
    t,i,h,w,s,two = flow.shape
    coords = get_img_coords(t,s,h,w).to(flow.device)
    coords = rearrange(coords,'two t 1 h w -> t 1 h w 1 two')
    pix_y,pix_x = flow[...,0],-flow[...,1]
    pix = torch.stack([pix_x,pix_y],dim=-1)
    pix += coords
    return pix

def pix2flow(pix):

    # -- pix2locs --
    t,i,h,w,s,two = pix.shape
    coords = get_img_coords(t,s,h,w).to(pix.device)
    coords = rearrange(coords,'two t 1 h w -> t 1 h w 1 two')
    locs = pix - coords

    # -- flip --
    flow_y = locs[...,0]
    flow_x = locs[...,1]
    flow = torch.stack([flow_x,-flow_y],dim=-1)

    return flow

def pix2locs(pix):
    # -- get coords --
    t,i,h,w,s,two = pix.shape
    coords = get_img_coords(t,s,h,w).to(pix.device)
    coords = rearrange(coords,'two t 1 h w -> t 1 h w 1 two')
    # loc_index_names(1,h,w,k,pix.device)

    # -- add --
    locs = pix - coords

    return locs

def locs2pix(locs):
    # -- get coords --
    t,i,h,w,s,two = locs.shape
    coords = get_img_coords(t,s,h,w).to(locs.device)
    coords = rearrange(coords,'two t 1 h w -> t 1 h w 1 two')
    # loc_index_names(1,h,w,k,pix.device)

    # -- add --
    pix = locs + coords

    return pix

def flow2locs(flow):
    pix = flow2pix(flow)
    locs = locs2pix(pix)
    return locs

def locs2flow(locs):
    pix = locs2pix(locs)
    flow = pix2flow(pix)
    return flow

# ------------------------------
#
#    Warp Burst From Pix/Flow
#
# ------------------------------

def warp_burst_from_flow(burst,flow,pad=None):
    locs = flow2locs(flow)
    return warp_burst_from_locs(burst,locs,pad)

def warp_burst_from_pix(burst,pix,pad=None):
    ndim = burst.dim()
    if ndim == 5:
        return warp_burst_from_pix_5d_burst(burst,pix,pad)
    if ndim == 4:
        return warp_burst_from_pix_4d_burst(burst,pix)
    else:
        msg = f"Uknown warp_burst_from_pix for burst with dims {ndim}"
        raise NotImplemented(msg)

def warp_burst_from_pix_5d_burst(burst,pix,pad):

    # -- block ranges per pixel --
    assert burst.dim() == 5,"The image batch dim is included"
    nframes_b,nimages_b,c,h_b,w_b = burst.shape
    nframes,nimages,h,w,k,two = pix.shape
    nparticles = k
    pix = rearrange(pix,'t i h w p two -> p i (h w) t two')
    if pad is None: pad = 0
    assert nframes_b == nframes,"eq frames"
    assert nimages_b == nimages,"eq images"

    # -- create offsets --
    warps = []
    for p in range(nparticles):
        warped = align_from_pix(burst,pix[p],pad)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)

    return warps

def warp_burst_from_pix_4d_burst(burst,pix):

    # -- block ranges per pixel --
    two,nframes,nsearch,h,w = pix.shape
    assert two == 2,"Input shape starts with two."

    # -- pix 2 locs --
    pix = rearrange(pix,'two t s h w -> t 1 h w s two')
    locs = pix2locs(pix)
    locs = rearrange(locs,'t 1 h w s two -> two t s h w')

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
    assert two == 2,"Input shape starts with two."
    if isize is None:
        isize = edict({"h":h,"w":w})
    if isinstance(isize,int):
        pad = isize
        isize = edict({"h":h+pad,"w":w+pad})

    # -- burst shape --
    burst = rearrange(burst,'c t h w -> t 1 c h w')

    # -- locs 2 flow --
    locs = rearrange(locs,'two t s h w -> t 1 h w s two')
    flows = locs2flow(locs)
    flows = rearrange(flows,'t i h w p two -> p i (h w) t two')

    # -- create offsets --
    warps = []
    for b in range(nblocks):
        warped = align_from_flow(burst,flows[b],0,isize=isize)
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
        warped = align_from_flow(burst,flows[p],0,isize=isize)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)
    return warps
