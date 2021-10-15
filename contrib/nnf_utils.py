###########################################
# NNF GPU Functions
###########################################

import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from nnf_share import *

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from align.xforms import pix_to_flow,align_from_flow

def runNnfBurstRecursive(_burst, clean, patchsize, nblocks, isize, k = 1,
                         valMean = 0., blockLabels=None, ref_t=None):
    wburst = _burst
    wclean = clean
    niters = 2
    for i in range(niters):
        wburst = wburst.to(_burst.device)
        wclean = wclean.to(_burst.device)
        vals,locs = runNnfBurst(wburst, patchsize,
                                nblocks, k = 1,
                                valMean = valMean,
                                blockLabels=blockLabels)
        pix = rearrange(locs,'t i h w 1 two -> i (h w) t two')
        flow = pix_to_flow(pix)
        wburst = align_from_flow(wburst,flow,nblocks,isize=isize)
        wclean = align_from_flow(wclean,flow,nblocks,isize=isize)
    return vals,locs,wclean

def runNnfBurst(_burst, patchsize, nblocks, k = 1,
                valMean = 0., blockLabels=None, ref_t=None,
                img_shape = None):

    # -- setup res --
    res = faiss.StandardGpuResources()
    burst = _burst#.cpu().clone().numpy()
    blockLabels = None
    device = burst.device # 'cpu'
    is_torch = True

    # -- set padded images --
    nframes,nimages,c,h,w = burst.shape
    if img_shape is None: img_shape = (c,h,w)
    iC,iH,iW = img_shape
    if ref_t is None: ref_t = nframes // 2
    if is_torch: dtype = torch.int32
    else: dtype = np.int32
    blockLabels,_ = getBlockLabels(blockLabels,nblocks,dtype,device,is_torch)
    
    # -- create ref indices-- 
    npix = iH*iW
    locs_ref = np.c_[np.unravel_index(np.arange(npix),(iH,iW))]
    locs_ref = locs_ref.reshape(iH,iW,2)
    locs_ref = repeat(locs_ref,'h w two -> i h w k two',i=nimages,k=k)
    locs_ref = torch.IntTensor(locs_ref).to(device,non_blocking=True)

    # -- run nnf --
    valsImages,locsImages = [],[]
    for i in range(nimages):
        refImg = burst[ref_t,i]
        refImgPad = padImage(refImg,img_shape,patchsize,nblocks)    
        valsFrames,locsFrames = [],[]
        for t in range(nframes):
            if t == ref_t:
                vals_t = torch.zeros((iH,iW,k),device=device)
                locs_t = locs_ref[i]
            else:
                tgtImg = burst[t,i]
                vals_t,locs_t = runNnf(res, img_shape, refImg,
                                       tgtImg, None, None,
                                       patchsize, nblocks, k,
                                       valMean = 0., blockLabels=blockLabels)
                if not(torch.is_tensor(vals_t)):
                    vals_t = torch.FloatTensor(vals_t)
                    locs_t = torch.IntTensor(locs_t)
                vals_t = vals_t.clone()
                locs_t = locs_t.clone()
            valsFrames.append(vals_t)
            locsFrames.append(locs_t)
        valsFrames = torch.stack(valsFrames)
        locsFrames = torch.stack(locsFrames)
        valsImages.append(valsFrames)
        locsImages.append(locsFrames)
    vals = torch.stack(valsImages,dim=1).cpu()
    locs = torch.stack(locsImages,dim=1).cpu().type(torch.long)

    # -- convert (y,x) -> (x,y) --
    locs_y = locs[...,0]
    locs_x = locs[...,1]
    locs = torch.stack([locs_x,locs_y],dim=-1)

    # -- convert (offset) -> (pix coord) --
    # locs = locs[ref_t] - locs
    for t in range(nframes):
        if t == ref_t: continue
        locs[t] = locs[ref_t] + locs[t]

    return vals,locs

def runNnf(res, img_shape, refImg, tgtImg, vals, locs, patchsize, nblocks, k = 3, valMean = 0., blockLabels=None):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    ref : array_like
        Reference image, shape (c, h, w).
        `dtype` must be float32.
    target : array_like
        Target image, shape (c, h, w).
        `dtype` must be float32.
    k : int
        Number of nearest neighbors.
    patchsize : int
        Size of patch in a single direction, a total of patchsize**2 pixels
    nblocks : int
        Number of neighboring patches to search in each direction, a total of nblocks**2
    D : array_like, optional
        Output array for distances of the nearest neighbors, shape (height, width, k)
    I : array_like, optional
        Output array for the nearest neighbors field, shape (height, width, k, 2).
        The "flow" goes _from_ refernce _to_ the target image.

    Returns
    -------
    vals : array_like
        Distances of the nearest neighbors, shape (height, width, k)
    locs : array_like
        Labels of the nearest neighbors, shape (height, width, k, 2)
    """

    # -- prepare data --
    c, h, w = img_shape
    refImgPad = padImage(refImg,img_shape,patchsize,nblocks)
    tgtImgPad = padImage(tgtImg,img_shape,patchsize,nblocks)
    refImg_ptr,refImg_type = getImage(refImgPad)
    tgtImg_ptr,tgtImg_type = getImage(tgtImgPad)
    is_tensor = torch.is_tensor(refImg)
    device = get_optional_device(refImg)
    assert torch.is_tensor(refImg) == torch.is_tensor(tgtImg),"Both torch or numpy."
    assert tgtImg_type == refImg_type,"Only one type for both"
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor)
    _,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,device,is_tensor)

    # -- setup args --
    args = faiss.GpuNnfDistanceParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.ps = patchsize
    args.nblocks = nblocks
    args.valMean = valMean # noise level value offset of minimum
    args.dtype = refImg_type
    args.refImg = refImg_ptr
    args.targetImg = tgtImg_ptr
    args.blockLabels = blockLabels_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        # faiss.bfNnf(res, args)
        with using_stream(res):
            faiss.bfNnf(res, args)
    else:
        faiss.bfNnf(res, args)

    return vals, locs
