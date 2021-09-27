###########################################
# NNF GPU Functions
###########################################

import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from nnf_share import *


def rows_uniq_elems(a):
    a_sorted = torch.sort(a,axis=-1)
    return a[(a_sorted[...,1:] != a_sorted[...,:-1]).all(-1)]

def evalAtFlow(burst, flow, patchsize, nblocks, return_mode=False):
    """
    Evaluate the code at the provided flow value.

    """
    vals,locs = [],[]
    valMean = 0. # defined to be zero for this fxn!
    pad = patchsize//2 + nblocks
    nimages,nsamples,nframes,two = flow.shape
    nframes,nimages,c,h,w = burst.shape
    for i in range(nimages):

        # -- create block labels --
        blockLabels = rearrange(flow[i],'s t two -> t s two')
        blockLabels_Y = blockLabels[...,0]
        blockLabels_X = blockLabels[...,1]
        blockLabels = torch.stack([-blockLabels_X,blockLabels_Y],dim=-1)

        # -- get unique -- 
        blockLabels = rearrange(blockLabels,'t p two -> p (t two)')
        blockLabels = torch.unique(blockLabels,dim=0,sorted=False)
        blockLabels = rearrange(blockLabels,'l (t two) -> t l two',t=nframes)
        nlabels = blockLabels.shape[1]

        # -- evaluate at block labels --
        vals_i,locs_i = runBurstNnf(burst, patchsize,
                                    nblocks, k = nlabels,
                                    valMean = valMean,
                                    blockLabels=blockLabels,
                                    fmt=False)

        # -- get shape to remove boarder --
        one,vH,vW,vK = vals_i.shape
        ccH = slice(pad,h-pad)
        ccW = slice(pad,w-pad)
        # print(h//2,h//2+1)
        # ccH = slice(h//2+0,h//2+1)
        # ccW = slice(w//2+0,w//2+1)

        # ccH = slice(h-1,h)
        # ccW = slice(w-1,w)

        # -- remove boarder --
        vals_i = vals_i[:,ccH,ccW,:]
        locs_i = locs_i[:,:,ccH,ccW,:,:]
        vals_i = rearrange(vals_i,'i h w k -> i (h w) k').cpu()
        locs_i = rearrange(locs_i,'i t h w k two -> k i (h w) t two').cpu().long()

        # -- reshape & append --
        vals.append(vals_i[0])
        locs.append(locs_i[:,0])

    # -- prepare output --
    vals = torch.stack(vals,dim=-1)
    locs = torch.stack(locs,dim=0)
    

    if return_mode:

        vals_cpu = vals[...,0].cpu().numpy().ravel()
        vmd = np.median(vals_cpu)
        delta = np.abs(vals_cpu - vmd)
        print("vals_cpu.shape ",vals_cpu.shape)
        vals_cpu = vals_cpu[np.where(delta < 1e2)]
        vals_float = torch.FloatTensor(vals_cpu)
        median = torch.median(vals_float).item()
        # vals_long = torch.LongTensor(vals_cpu*1e4)
        # print("vals_long.shape ",vals_long.shape)
        # mode = torch.mode(vals_long).values.item() * 1e-4 
        mode = median
        return mode

    else:
        return vals,locs


def runBurstNnf(burst, patchsize, nblocks, k = 1,
                valMean = 0., blockLabels=None, ref=None,
                to_flow=False, fmt=False):

    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    img_shape = (c,h,w)
    if ref is None: ref = nframes//2

    # -- compute search "blockLabels" across image burst --
    vals,locs = [],[]
    for i in range(nimages):
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)
        vals_i,locs_i = _runBurstNnf(res, img_shape, burstPad_i,
                                     ref, None, None,
                                     patchsize, nblocks,
                                     k = k, valMean = valMean,
                                     blockLabels=blockLabels)
        vals.append(vals_i)
        locs.append(locs_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=0)
    # (nimages, nframes, h, w, k, two)

    if to_flow:
        locs_y = locs[...,0]
        locs_x = locs[...,1]
        locs = torch.stack([locs_x,-locs_y],dim=-1)
    
    if fmt:
        vals = rearrange(vals,'i h w k -> i (h w) k').cpu()
        locs = rearrange(locs,'i t h w k two -> k i (h w) t two').cpu().long()

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

    # print("bl")
    # print("-"*50)
    # for i in range(bl.shape[1]):
    #     print(bl[:,i,:].cpu().numpy())
    # print(bl.shape)
    # print("-"*50)
    # print("bl")
    
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
    args.nblocks_total = bl.shape[1]
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
