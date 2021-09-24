###########################################
# NNF GPU Functions
###########################################

import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from nnf_share import *


def runBurstNnf(burst, patchsize, nblocks, k = 1,
                valMean = 0., blockLabels=None, ref=None):

    res = faiss.StandardGpuResources()
    nframes,nimages,c,h,w = burst.shape
    img_shape = (c,h,w)
    if ref is None: ref = nframes//2
    vals,locs = [],[]
    for i in range(nimages):
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)
        vals_i,locs_i = _runBurstNnf(res, img_shape, burstPad_i,
                                     ref, None, None,
                                     patchsize, nblocks,
                                     k = k, valMean = valMean,
                                     blockLabels=None)
        vals.append(vals_i)
        locs.append(locs_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=0)
    # (nimages, nframes, h, w, k, two)

    return vals,locs

def _runBurstNnf(res, img_shape, burst, ref, vals, locs, patchsize, nblocks, k = 3, valMean = 0., blockLabels=None):
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
    nframes = burst.shape[0]
    burstPad = padBurst(burst,img_shape,patchsize,nblocks)
    burst_ptr,burst_type = getImage(burstPad)
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,nframes)
    bl,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,
                                       device,is_tensor,nframes)

    print("bl")
    print(bl)
    print(bl.shape)

    # -- setup args --
    args = faiss.GpuBurstNnfDistanceParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.t = nframes
    args.ps = patchsize
    args.nblocks = nblocks
    args.valMean = valMean # noise level value offset of minimum
    args.burst = burst_ptr
    args.dtype = burst_type
    args.blockLabels = blockLabels_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        with using_stream(res):
            faiss.bfBurstNnf(res, args)
    else:
        faiss.bfBurstNnf(res, args)

    return vals, locs
