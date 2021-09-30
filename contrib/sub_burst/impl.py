
import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from .utils import *

def _runBurstNnf(res, img_shape, total_nframes, burst, subAve, vals, locs,
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
    burst_ptr,burst_type = getImage(burstPad)
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,sub_nframes)
    subAve,subAve_ptr,subAve_type = getSubAveTorch(subAve,hP,wP,c,device,t=None)
    bl,blockLabels_ptr = getBlockLabelsFull(blockLabels,img_shape,nblocks,locs.dtype,
                                            device,is_tensor,sub_nframes)
    # bl.shape = nsearch,h,w,nframes,two
    nsearch = bl.shape[0]
    mask,mask_ptr = getMask(nsearch,h,w,sub_nframes,device,is_tensor)

    assert bl.dim() == 5,"5 dimensional block labels"


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
