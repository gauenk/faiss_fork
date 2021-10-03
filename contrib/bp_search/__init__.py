"""
Belief Propogation Search

"""

import torch
import faiss
import numpy as np
import torchvision
from einops import rearrange,repeat
import nnf_utils as nnf_utils
from nnf_share import padBurst,getBlockLabels,tileBurst,padAndTileBatch
# from bnnf_utils import runBurstNnf
from sub_burst import runBurstNnf as runSubBurstNnf
# from wnnf_utils import runWeightedBurstNnf

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")

from .utils import create_search_ranges,warp_burst,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs
from .merge_search_ranges_numba import merge_search_ranges

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def padLocs(locs,pad):
    nframes,nimages,h,w,k,two = locs.shape
    locs = rearrange(locs,'t i h w k two -> t i k two h w')
    plocs = th_pad(locs,(pad,)*4)
    plocs = rearrange(plocs,'t i k two h w -> t i h w k two')
    return plocs

def runBpSearch(burst, patchsize, nblocks, k = 1,
                nparticles = 1, niters = 2,
                valMean = 0.,
                l2_nblocks = 3, l2_valMean=0.,
                blockLabels=None, ref=None,
                to_flow=False, fmt=False):

    assert nparticles == 1, "Only one particle currently supported."

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = burst.device
    nframes,nimages,c,h,w = burst.shape
    img_shape = [c,h,w]
    pad = 2*(nblocks//2)
    ishape = [h,w]
    pshape = [h+pad,w+pad]
    def ave_denoiser(wburst,mask,nframes_per_pix):
        return torch.sum(wburst*mask,dim=0) / nframes_per_pix

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
    print("l2_nblocks: ",l2_nblocks)
    vals,pix = nnf_utils.runNnfBurst(burst, patchsize, l2_nblocks,
                                      k=nparticles, valMean = l2_valMean,
                                      img_shape = None)
    vals = torch.mean(vals,dim=0).to(device)
    print("pix.shape ",pix.shape)
    pix = pix.to(device)
    locs = pix2locs(pix)
    
    print(locs[:,0,15,16,:])
    print(locs[:,0,16,15,:])
    print(locs[:,0,16,16,:])
    print("locs.shape: ",locs.shape)

    # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)
    search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)

    # -- 3.) warp burst to top location --
    print("burst.shape ",burst.shape)
    wburst = padAndTileBatch(burst,patchsize,nblocks)
    ppix = padLocs(pix,patchsize//2)
    print("wburst.shape ",wburst.shape)
    warped_burst = warp_burst(wburst,ppix,nblocks)
    print("warped_burst.shape ",warped_burst.shape)
    img_shape[0] = wburst.shape[-3]


    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    clK = [3,]*niters # -- scheduler for clustering --
    # search_blocks = compute_search_blocks(search_ranges,3) # -- matches clK --
    # K = nframes

    for i in range(niters):

        # -- 1.) cluster each pixel across time --
        K = clK[i]
        # print("wburst.shape: ",wburst.shape)
        print("[start] compute_temporal_cluster")
        cc_wburst = center_crop(warped_burst,pshape)
        names,means,weights,mask = compute_temporal_cluster(cc_wburst,K)
        wmeans = means * weights
        wmeans = wmeans[0] # nparticles == 1
        wmeans = wmeans.contiguous()
        cc_names = center_crop(names,ishape)
        print("[end] compute_temporal_cluster")

        # print("search_ranges.shape ",search_ranges.shape)
        # print("msr.shape ",msr.shape)
        # locs = split_locs(best_locs,offsets,cc_names,msr,nblocks)
        # locs_frames2groups(locs,cc_names,search_ranges,nblocks)
        # print(glocs)
        # print(glocs.shape)
        # exit()

        # # -- 2.) denoise each group (e.g. use averages) --
        # dclusters = denoise_clustered_burst(wburst,clusters,ave_denoiser)
        # -- skip for now --

        # -- 3.) create combinatorial search blocks from search ranges  --
        print("[start] merge_search_ranges")
        refG = 1
        merged_search_ranges = merge_search_ranges(pix,cc_names,search_ranges,
                                                   nblocks,pixAreLocs=False)
        search_blocks = compute_search_blocks(merged_search_ranges,refG)
        print("search_blocks.shape: ",search_blocks.shape)
        print("[end] merge_search_ranges")
        print("Extrema of Search Blocks: ",search_blocks.max(),search_blocks.min())
        
        # -- 4.) exh srch over a clusters --
        in_burst = wmeans
        # in_burst = warped_burst[0]
        # in_burst = wburst
        print("in_burst.shape: ",in_burst.shape)
        sub_vals,sub_locs = runSubBurstNnf(in_burst,1,
                                           nblocks,k=1, # patchsize=1 since tiled
                                           mask=mask,
                                           blockLabels=search_blocks,
                                           img_shape = img_shape)
        print("vals: ",vals.shape)
        print("locs: ",locs.shape)
        print("sub_vals: ",sub_vals.shape)
        print("sub_locs: ",sub_locs.shape)
        sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')
        print("[post] sub_locs: ",sub_locs.shape)
        print(vals[:,16,16])
        print(sub_vals[:,16,16])
        vals = sub_vals
        locs = sub_locs + locs
        print("iter: ",i)
        print(sub_locs[:,:,16,16])
        
        # print("names: ",names.shape)
        # print(vals[...,32,32,0])
        # sub_vals,sub_locs = runWeightedBurstNnf(means,weights,patchsize,nblocks,
        #                                         blockLabels=search_blocks)

        # -- 5.) update vals and locs --
        # overwrite = i == 0
        # names = names[0] # nparticles == 1
        # cc_names = center_crop(names,img_shape[1:])
        # vals,locs = update_state(vals,locs,sub_vals,sub_locs,cc_names,overwrite)

        # -- 6.) rewarp bursts --
        # print("locs.shape ",locs.shape)
        # cc_wburst = center_crop(wburst,img_shape[1:])
        # print(cc_wburst.shape)
        # hP,wP = wburst.shape[-2:]
        # wlocs = rearrange(locs,'t i h w k two -> t i k two h w')
        # wlocs = th_pad(wlocs,(pad,pad),0,'constant')
        # wlocs = rearrange(wlocs,'t i k two h w -> t i h w k two')
        # print("wlocs.shape ",wlocs.shape)
        # wburst = warp_burst(wburst[0],wlocs,nblocks)
        # print(wburst.shape)
        # wburst = padAndTileBatch(wburst[0],patchsize,nblocks)[None,:]
        # break


    return vals,locs
