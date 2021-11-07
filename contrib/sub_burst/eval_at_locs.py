import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from nnf_share import *
from .run_burst import runBurstNnf

def evalAtLocs(burst, locs, patchsize, nblocks,
               return_mode = False, img_shape = None):
    """
    Evaluate the code at the provided loc (offsets) value.

    """
    valMean = 0. # defined to be zero for this fxn!
    # pad = patchsize//2 + nblocks
    nframes,nimages,h,w,nsamples,two = locs.shape
    nframes,nimages,c,h,w = burst.shape

    if not(img_shape is None):
        c,h,w = img_shape
    else:
        c,h,w = burst.shape[-3:]
        img_shape = [c,h,w]

    # -- pad locs --
    # print("-"*30)
    # print(locs[:,0,h//2,w//2,0,:])
    # print("-"*30)
    hL,wL,_,_ = locs.shape[-4:]
    nbHalf = nblocks//2
    psHalf = patchsize//2
    pad = nbHalf + psHalf
    if ((h+2*pad != hL) or (w+2*pad != wL)):
        pad_locs = padLocs(locs,pad,'extend')
    else:
        pad_locs = locs
        
    # print("img_shape: ",img_shape)
    # print("burst.shape :",burst.shape)
    # print("pad_locs.shape :",pad_locs.shape)

    # -- evaluate at block labels --
    pad_locs = rearrange(pad_locs,'t i h w s two -> s i h w t two')
    out_vals,out_locs = runBurstNnf(burst, patchsize,
                                    nblocks, k = nsamples,
                                    valMean = valMean,
                                    blockLabels=pad_locs,
                                    fmt=False,
                                    img_shape = img_shape)
    
    # -- possible reshaping --
    # vals_i = rearrange(vals_i,'i h w k -> i (h w) k').cpu()
    # locs_i = rearrange(locs_i,'i t h w k two -> k i (h w) t two').cpu().long()

    return out_vals,out_locs
