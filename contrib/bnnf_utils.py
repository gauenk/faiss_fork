###########################################
# NNF GPU Functions
###########################################

import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import using_stream
from nnf_share import *
# import nnf_utils

def evalAtFlow(burst, flow, patchsize, nblocks,
               return_mode=False, tile_burst=False,
               img_shape=None):
    """
    Evaluate the code at the provided flow value.

    """

    vals,locs = [],[]
    valMean = 0. # defined to be zero for this fxn!
    pad = patchsize//2 + nblocks//2
    nimages,nsamples,nframes,two = flow.shape
    if nsamples > 1:
        print("WARNING! Only works for a one flow used at EACH pixel location.")
        flow = flow[:,[0]]
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
                                    fmt=False,tile_burst=tile_burst,
                                    img_shape=img_shape)

        # -- get shape to remove boarder --
        one,vH,vW,vK = vals_i.shape
        # ccH = slice(pad,h-pad)
        # ccW = slice(pad,w-pad)
        # print(h//2,h//2+1)
        # ccH = slice(1,2)
        # ccW = slice(1,2)
        # ccH = slice(0,1000)
        # ccW = slice(0,1000)

        # ccH = slice(h-1,h)
        # ccW = slice(w-1,w)

        # -- remove boarder --
        # vals_i = vals_i[:,ccH,ccW,:]
        # locs_i = locs_i[:,:,ccH,ccW,:,:]
        if return_mode:
            vals_i = rearrange(vals_i,'i h w k -> i (h w) k').cpu()
            locs_i = rearrange(locs_i,'i t h w k two -> i k (h w) t two').cpu().long()

        # -- reshape & append --
        vals.append(vals_i[0])
        locs.append(locs_i[0])

    # -- prepare output --

    if return_mode:

        vals = torch.stack(vals,dim=-1)
        locs = torch.stack(locs,dim=0)

        # print(vals.shape)
        h = int(np.sqrt(vals.shape[0]))
        vals_cpu = vals[...,0].cpu().numpy()#.ravel()
        vals_cpu = rearrange(vals_cpu,'(h w) 1 -> h w',h=h)
        vmd = np.median(vals_cpu)
        delta = np.abs(vals_cpu - vmd)
        # vals_cpu = vals_cpu[np.where(delta < 1e2)]
        vals_float = torch.FloatTensor(vals_cpu)
        median = vals_float
        # median = torch.median(vals_float).item()
        # vals_long = torch.LongTensor(vals_cpu*1e4)
        # print("vals_long.shape ",vals_long.shape)
        # mode = torch.mode(vals_long).values.item() * 1e-4 
        mode = median
        return mode

    else:

        vals = torch.stack(vals,dim=0)
        locs = torch.stack(locs,dim=0)

        return vals,locs

def runBurstNnf(burst, patchsize, nblocks, k = 1,
                valMean = 0., blockLabels=None, ref=None,
                to_flow=False, fmt=False, in_vals=None,in_locs=None,
                tile_burst=False,img_shape=None):

    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    if img_shape is None: img_shape = [c,h,w]
    device = burst.device
    if ref is None: ref = nframes//2

    # # -- get block labels once for the burst if "None" --
    # blockLabels,_ = getBlockLabels(blockLabels,nblocks,torch.long,
    #                                device,True,nframes)

    # -- compute search "blockLabels" across image burst --
    vals,locs = [],[]
    for i in range(nimages):

        # -- create padded burst --
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)
        # print("[bnnf_utils, pre]: burstPad_i.shape ",burstPad_i.shape)
        if tile_burst:
            burstPad_i = tileBurst(burstPad_i,h,w,patchsize,nblocks)
            img_shape = list(img_shape)
            img_shape[0] = burstPad_i.shape[1]
            input_ps = 1
        else:
            input_ps = patchsize

        # print("[bnnf_utils, post]: burstPad_i.shape ",burstPad_i.shape)
        # -- assign input vals and locs --
        vals_i,locs_i = in_vals,in_locs
        if not(in_vals is None): vals_i = vals_i[i]
        if not(in_locs is None): locs_i = locs_i[i]

        # -- execute over search space! --
        vals_i,locs_i = _runBurstNnf(res, img_shape, burstPad_i,
                                     ref, vals_i, locs_i,
                                     input_ps, nblocks,
                                     k = k, valMean = valMean,
                                     blockLabels = blockLabels)

        vals.append(vals_i)
        locs.append(locs_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=1)
    # (nframes, nimages, h, w, k, two)

    if to_flow:
        locs_y = locs[...,0]
        locs_x = locs[...,1]
        locs = torch.stack([locs_x,-locs_y],dim=-1)
    
    if fmt:
        vals = rearrange(vals,'i h w k -> i (h w) k').cpu()
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two').cpu().long()

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
    # print("[bnnf_utils._runBurstNnf, pre]: burst.shape ",burst.shape)
    burstPad = padBurst(burst,img_shape,patchsize,nblocks)
    # print("[bnnf_utils._runBurstNnf, post]: burstPad.shape ",burstPad.shape)
    burst_ptr,burst_type = getImage(burstPad)
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,nframes)
    bl,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,
                                       device,is_tensor,nframes)
    # print("[bnnf_utils]: burstPad.shape ",burstPad.shape)
    # print("[bnnf_utils]: bl.shape ",bl.shape)

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
