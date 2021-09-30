import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from nnf_share import *
from .run_burst import runBurstNnf

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
        # ccH = slice(pad,h-pad)
        # ccW = slice(pad,w-pad)
        # print(h//2,h//2+1)
        ccH = slice(h//2+0,h//2+1)
        ccW = slice(w//2+0,w//2+1)

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
