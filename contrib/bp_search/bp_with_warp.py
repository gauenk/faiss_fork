"""
Belief Propogation Search


The best location can be shuffled
_away_ from a specific patch becuase
the warping is conditioned on the current state.

Said another way...

For a specific state S, the 
resulting warped image, I | S,
can move the best neighboring patch
_further away_ and it is unclear
if there is a way for the best
patch to "come back" into reach for
a specific patch.


"""

import torch
import faiss
import numpy as np
import torchvision
from einops import rearrange,repeat
import nnf_utils as nnf_utils
from nnf_share import padBurst,getBlockLabels,tileBurst,padAndTileBatch,padLocs,locs2flow
# from bnnf_utils import runBurstNnf
from sub_burst import runBurstNnf as runSubBurstNnf
from sub_burst import evalAtLocs
# from wnnf_utils import runWeightedBurstNnf
from easydict import EasyDict as edict

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")

from .utils import create_search_ranges,warp_burst_from_pix,warp_burst_from_locs,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs,index_along_ftrs
from .merge_search_ranges_numba import merge_search_ranges

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def runBpSearch(noisy, clean, patchsize, nblocks, k = 1,
                nparticles = 1, niters = 20,
                valMean = 0.,
                l2_nblocks = None, l2_valMean=0.,
                blockLabels=None, ref=None,
                to_flow=False, fmt=False):

    if l2_nblocks is None: l2_nblocks = nblocks
    assert nparticles == 1, "Only one particle currently supported."

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = noisy.device
    nframes,nimages,c,h,w = noisy.shape
    img_shape = [c,h,w]
    pad = 2*(nblocks//2)
    ishape = [h,w]
    isize = edict({'h':h,'w':w})
    pisize = edict({'h':h+pad,'w':w+pad})
    pshape = [h+pad,w+pad]
    def ave_denoiser(wnoisy,mask,nframes_per_pix):
        return torch.sum(wnoisy*mask,dim=0) / nframes_per_pix

    # -------------------------------
    #
    # ----    inital search      ----
    #
    #  -> large patchsize to account
    #      for large global motion.
    #
    #  -> search over keypoints only
    #
    # -------------------------------

    # -- 1.) run l2 local search --
    vals,pix = nnf_utils.runNnfBurst(noisy, patchsize, l2_nblocks,
                                      k=nparticles, valMean = l2_valMean,
                                      img_shape = None)
    vals = torch.mean(vals,dim=0).to(device)
    l2_vals = vals
    pix = pix.to(device)
    locs = pix2locs(pix)
    l2_locs = locs

    # # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)
    search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)

    # -- 3.) warp burst to top location --
    wnoisy = padAndTileBatch(noisy,patchsize,nblocks)
    wclean = padAndTileBatch(clean,patchsize,nblocks)
    pixPad = (wnoisy.shape[-1] - noisy.shape[-1])//2
    ppix = padLocs(pix+pixPad,pixPad)
    plocs = padLocs(locs,pixPad)

    warped_noisy = warp_burst_from_locs(wnoisy,plocs,1,pisize)
    warped_clean = warp_burst_from_locs(wclean,plocs,1,pisize)
    img_shape[0] = wnoisy.shape[-3]
    
    hP,wP = h + 2*pixPad,w + 2*pixPad
    pad_search_ranges = create_search_ranges(nblocks,hP,wP,nframes)
    pad_search_ranges = torch.LongTensor(pad_search_ranges[:,None]).to(device)


    # -------------------------------
    #
    #   Correctly set values using  
    #   Sub-Ave instead of L2
    #
    # -------------------------------
    sub_vals = l2_vals
    sub_locs_rs = l2_locs
    psHalf = patchsize//2
    vals,e_locs = evalAtLocs(wnoisy,
                             sub_locs_rs, 1,
                             nblocks,
                             img_shape=img_shape)

    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    sub_locs_rs_pad = padLocs(sub_locs_rs,psHalf,mode='extend')
    psHalf = patchsize//2
    nbHalf = nblocks//2
    padSize = psHalf

    clK = [3,]*niters#niters # -- scheduler for clustering --
    # search_blocks = compute_search_blocks(search_ranges,3) # -- matches clK --
    # K = nframes

    exp_locs = locs
    for i in range(niters):

        # -- 1.) cluster each pixel across time --
        K = clK[i]
        names,means,weights,mask = compute_temporal_cluster(warped_noisy,K)
        wmeans = means * weights/nframes
        wmeans = wmeans[0] # nparticles == 1
        wmeans = wmeans.contiguous()
        cc_names = center_crop(names,ishape)

        # # -- 2.) denoise each group (e.g. use averages) --
        # dclusters = denoise_clustered_noisy(wnoisy,clusters,ave_denoiser)
        # -- skip for now --

        # -- 3.) create combinatorial search blocks from search ranges  --
        refG = nframes//2
        pad_locs = padLocs(locs,nbHalf,'extend') # since patchsize is 1 here.
        merged_search_ranges,offsets = merge_search_ranges(pad_locs,names,
                                                           pad_search_ranges,
                                                           nblocks,pixAreLocs=True)

        # -- 4.) search space --
        search_blocks = compute_search_blocks(merged_search_ranges,refG)
        search_blocks = rearrange(search_blocks,'l i h w t two -> l i t two h w')
        search_blocks = center_crop(search_blocks,img_shape[1:])
        search_blocks = rearrange(search_blocks,'l i t two h w -> l i h w t two')
        search_blocks = search_blocks.contiguous()

        # -- 5.) exh srch over a clusters --
        sub_vals,sub_locs = runSubBurstNnf(wmeans,1,
                                           nblocks,k=1, # patchsize=1 since tiled
                                           mask=mask,
                                           blockLabels=search_blocks,
                                           img_shape = img_shape)
        sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')

        # -- 6.) update vals and locs --
        cc_names = center_crop(names[...,0,:,:],img_shape[1:])
        vals,locs,exp_locs = update_state(vals,locs,sub_vals,sub_locs,cc_names,False)
        max_displ = np.max([locs.max().item(),np.abs(locs.min().item())])
        assert max_displ <= nblocks//2, "displacement must remain contained!"


        # -- 7.) rewarp bursts --
        pad_locs = padLocs(locs,nbHalf,'extend')
        nframes,nimages,hP,wP,k,two = pad_locs.shape
        psize = edict({'h':hP,'w':wP})
        p_exp_locs = padLocs(exp_locs,nbHalf,'extend')
        warped_noisy = warp_burst_from_locs(warped_noisy[0],p_exp_locs,nblocks,psize)
        warped_clean = warp_burst_from_locs(warped_clean[0],p_exp_locs,nblocks,psize)

    warped_noisy = center_crop(warped_noisy[0],ishape)
    warped_clean = center_crop(warped_clean[0],ishape)

    warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    warped_clean = index_along_ftrs(warped_clean,patchsize,c)
    
    if to_flow:
        locs = locs2flow(locs)

    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs,warped_noisy,warped_clean
