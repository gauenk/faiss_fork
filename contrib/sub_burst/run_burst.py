
import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from nnf_share import padBurst
from .impl import _runBurstNnf

def runBurstNnf(burst, patchsize, nblocks, k = 1,
                valMean = 0., subAve=None,
                mask = None,
                tframes = None,
                blockLabels=None, ref=None,
                img_shape = None,
                to_flow=False, fmt=False, in_vals=None,in_locs=None):

    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    if img_shape is None: img_shape = (c,h,w)
    if ref is None: ref = nframes//2
    if tframes is None: total_nframes = nframes
    else: total_nframes = tframes

    # -- compute search "blockLabels" across image burst --
    vals,locs = [],[]
    for i in range(nimages):

        # -- create padded burst --
        # print("[subBurst.run_burst]: burst[:,i].shape ",burst[:,i].shape)
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)
        # print("[subBurst.run_burst]: burstPad_i.shape ",burstPad_i.shape)

        # -- assign input vals and locs --
        vals_i,locs_i = in_vals,in_locs
        if not(in_vals is None): vals_i = vals_i[i]
        if not(in_locs is None): locs_i = locs_i[i]
        if not(mask is None): mask_i = mask[i]
        else: mask_i = None
        if not(blockLabels is None): search_space_i = blockLabels[:,i]
        else: search_space_i = None

        # -- execute over search space! --
        vals_i,locs_i = _runBurstNnf(res, img_shape,
                                     total_nframes,
                                     burstPad_i,
                                     subAve, mask_i,
                                     vals_i, locs_i,
                                     patchsize, nblocks,
                                     k = k, valMean = valMean,
                                     blockLabels = search_space_i)
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
