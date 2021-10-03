
import torch
import torchvision
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from .utils import *

def _runBurstNnf(res, img_shape, total_nframes, burst, subAve, mask, vals, locs,
                 patchsize, nblocks, k = 3, valMean = 0., blockLabels=None):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    ref : array_like
        Burst of images, shape (t, c, h, w).
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
    burstPad = padBurst(burst,img_shape,patchsize,nblocks)
    sub_nframes,c,hP,wP = burstPad.shape
    hsP = hP - 2*(nblocks//2)
    wsP = wP - 2*(nblocks//2)
    burst_ptr,burst_type = getImage(burstPad)
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,sub_nframes)
    subAve,subAve_ptr,subAve_type = getSubAveTorch(subAve,hsP,wsP,c,device,t=None)
    img_shape = (c,hsP,wsP)
    bl,blockLabels_ptr = getBlockLabelsFull(blockLabels,img_shape,nblocks,locs.dtype,
                                            device,is_tensor,sub_nframes)
    bl = rearrange(bl,'l h w t two -> l t two h w')
    # pad = (nblocks//2) + (patchsize//2)
    pad = (patchsize//2)
    print("bl.shape ",bl.shape)
    # bl = torchvision.transforms.functional.pad(bl,(pad,)*4)
    print("bl.shape ",bl.shape)
    bl = rearrange(bl,'l t two h w -> l h w t two')
    bl = bl.contiguous()
    blockLabels_ptr,_ = torch2swig(bl)

    nsearch = bl.shape[0]
    if mask is None:
        mask,mask_ptr = getMask(nsearch,h,w,
                                sub_nframes,device,is_tensor)
    else:
        mask_ptr,_ = torch2swig(mask)

    assert bl.dim() == 5,"5 dimensional block labels"


    print("[subBurst]: burst.shape ",burst.shape)
    print("[subBurst]: burstPad.shape ",burstPad.shape)
    print("vals.shape ",vals.shape)
    print("locs.shape ",locs.shape)
    print("subAve.shape ",subAve.shape)
    print("masks.shape ",mask.shape)
    print("bl.shape ",bl.shape)


    # print("bl")
    # print("-"*50)
    # for i in range(bl.shape[1]):
    #     print(bl[:,i,:].cpu().numpy())
    # print(bl.shape)
    # print("-"*50)
    # print("bl")
    
    # -- setup args --
    args = faiss.GpuSubBurstNnfDistanceParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.sub_t = sub_nframes
    args.t = total_nframes
    args.ps = patchsize
    args.nblocks = nblocks
    args.nblocks_total = bl.shape[0]
    args.valMean = valMean # noise level value offset of minimum
    args.burst = burst_ptr
    args.subAve = subAve_ptr
    args.mask = mask_ptr
    args.dtype = burst_type
    args.blockLabels = blockLabels_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        with using_stream(res):
            faiss.bfSubBurstNnf(res, args)
    else:
        faiss.bfSubBurstNnf(res, args)

    return vals, locs
