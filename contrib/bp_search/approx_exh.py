"""
Belief Propogation Search

"""

import torch
import faiss
import numpy as np
import torchvision
from einops import rearrange,repeat
import nnf_utils as nnf_utils
from nnf_share import padBurst,getBlockLabels,tileBurst,padAndTileBatch,padLocs,locs2flow,mode_vals,mode_ndarray
from bnnf_utils import runBurstNnf,evalAtFlow
from sub_burst import runBurstNnf as runSubBurstNnf
from sub_burst import evalAtLocs
# from wnnf_utils import runWeightedBurstNnf
from easydict import EasyDict as edict

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")

from .utils import create_search_ranges,warp_burst_from_pix,warp_burst_from_locs,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs,index_along_ftrs,temporal_inliers_outliers,update_state_outliers
from .merge_search_ranges_numba import merge_search_ranges

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def center_crop_locs(locs,hw_list):
    locs = rearrange(locs,'t i h w k two -> t i k two h w')
    locs = center_crop(locs,hw_list)
    locs = rearrange(locs,'t i k two h w -> t i h w k two')
    return locs

def runBpSearchApproxExh(noisy, patchsize, nblocks, k = 1,
                         nparticles = 1, niters = 5,
                         valMean = 0., std = None,
                         l2_nblocks = None, l2_valMean=0.,
                         blockLabels=None,
                         search_ranges=None, ref=None,
                         to_flow=False, fmt=False,
                         img_shape = None):


    # -------------------------------
    #
    # ----    error checking     ----
    #
    # -------------------------------

    if l2_nblocks is None: l2_nblocks = nblocks
    assert nparticles == 1, "Only one particle currently supported."

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = noisy.device
    nframes,nimages,c,h,w = noisy.shape
    input_img_shape = img_shape
    if img_shape is None: img_shape = [c,h,w]
    c,h,w = img_shape
    ishape = [h,w]
    psHalf,nbHalf = patchsize//2,nblocks//2
    fPad = 2*(psHalf + nbHalf)
    int_shape = [h-fPad,w-fPad]
    isize = edict({'h':h,'w':w})
    psize = edict({'h':h+2*nbHalf,'w':w+2*nbHalf})
    pshape = [h+psHalf,w+psHalf]
    mask = torch.zeros(h+2*nbHalf,w+2*nbHalf).to(device)
    MAX_SEARCH_FRAMES = 3
    numSearch = min(MAX_SEARCH_FRAMES,nframes-1)
    if std is None: std = torch.std(noisy.reshape(-1)).item()
    if np.isclose(valMean,0):
        p = patchsize**2
        t = numSearch + 1
        c2 = ((t-1)/t)**2 * std**2 + (t-1)/t**2 * std**2
        mode = c2*(1 - 2/p)*p
        if mode < 0:
            print("Mode was less than zero.")
            mode = 0 # this happens if patchsize == 1 after tiling.
        valMean = mode

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
    assert torch.any(torch.isnan(noisy)).item() is False, "no nans plz."
    vals,pix = nnf_utils.runNnfBurst(noisy, patchsize, l2_nblocks,
                                      k=nparticles, valMean = l2_valMean,
                                      img_shape = input_img_shape)
    vals = torch.mean(vals,dim=0).to(device)
    l2_vals = vals
    pix = pix.to(device)
    locs = pix2locs(pix)
    locs = torch.zeros_like(locs)
    l2_locs = torch.zeros_like(locs)
    # l2_locs = locs
    vals[...] = 1000.

    # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    sRefG = ref
    if search_ranges is None:
        search_ranges = create_search_ranges(nblocks,h,w,nframes)
        search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)
        sRefG = nframes//2

    # -- 3.) pad and tile --
    eqH = noisy.shape[-2] == (img_shape[-2]+2*nbHalf)
    eqW = noisy.shape[-1] == (img_shape[-1]+2*nbHalf)
    # print("noisy.shape: ",noisy.shape)
    # print(eqH,eqW)
    # print(img_shape)
    if not(eqH and eqW):
        tnoisy = padAndTileBatch(noisy,patchsize,nblocks)
        img_shape[0] = tnoisy.shape[-3]
    else:
        tnoisy = noisy

    # -- 4.) update "val" from "l2" to "burst" @ curr --
    vals,e_locs = evalAtLocs(tnoisy, l2_locs, 1,
                             nblocks, img_shape=img_shape)

    # -- 5.) warp burst to top location --
    pixPad = (tnoisy.shape[-1] - img_shape[-1])//2
    plocs = padLocs(locs,pixPad,'extend')
    warped_noisy = warp_burst_from_locs(tnoisy,plocs,1,psize)[0]

    # -- compute search ranges for number of search frames --
    ngroups = numSearch+1
    refG = ref#ngroups//2-1
    mid = ngroups//2
    # print("[approx_exh] ",refG,ngroups,ref)
    if blockLabels is None:

        search_blocks,_ = getBlockLabels(None,nblocks,torch.long,device,True,t=ngroups)

        # -- swap to ref --
        # tmp = search_blocks[mid].clone()
        # # print(tmp)
        # assert torch.all(tmp == 0).item() is True ,"all zero plz"
        # search_blocks[mid] = search_blocks[refG]
        # search_blocks[refG] = tmp
        
        # -- put zeros @ start since ref is always 1st --
        ngroups,nsearch,two = search_blocks.shape
        left = search_blocks[:mid] 
        right = search_blocks[mid+1:] 
        search_blocks = torch.cat([search_blocks[[mid]],left,right],dim=0)
        # print("None!")
        search_blocks = repeat(search_blocks,'t l two -> l i h w t two',
                               i=nimages,h=h,w=w)
    else:
        # search_blocks = rearrange(blockLabels[:,0,16,16,:,:],'l t two -> t l two')
        search_blocks = rearrange(blockLabels,'l i h w t two -> t i h w l two')
        left = search_blocks[:refG]
        right = search_blocks[refG+1:] 
        search_blocks = torch.cat([search_blocks[[refG]],left,right],dim=0)
        search_blocks = rearrange(search_blocks,'t i h w l two -> l i h w t two')
    # print("[approx_exh] search_blocks.shape: ",search_blocks.shape)
    # print(search_blocks[0])
    assert torch.all(search_blocks[...,0,:] == 0).item() is True, "all zero @ ref."
    # assert torch.all(search_blocks[0] == 0).item() is True, "all zero @ ref."

    
    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    counts = torch.zeros(nframes)
    for i in range(niters):
        prev_locs = locs.clone()
        # print("numSearch: ",numSearch)

        # -- 1.) cluster each pixel across time --
        search_frames,names,nuniuqes = temporal_inliers_outliers(tnoisy,warped_noisy,
                                                                 vals,std,
                                                                 numSearch=numSearch,
                                                                 ref=refG)
        # print("names: ",names) # changes choices and order each iteration, ref #1 always
        counts[names] += 1

        # -- 1.ii) create search space from cluster --
        # merged_search_ranges,offsets = merge_search_ranges(pad_locs,names,
        #                                                    pad_search_ranges,
        #                                                    nblocks,pixAreLocs=True,
        #                                                    drift=True)
        # print("search_ranges.shape: ",search_ranges.shape)
        msr = [search_ranges[...,sRefG,:]]
        msr += [search_ranges[...,t,:] for t in names]
        msr = torch.stack(msr,dim=-2)
        # print("msr.shape: ",msr.shape)
        assert torch.all(msr[...,0,:] == 0).item() is True, "all zero @ ref."
        search_blocks = compute_search_blocks(msr,0)

        # -- 2.) exh srch over a selected frames --
        # print("-"*50)
        # print("Exh Search.")
        # print("-"*50)
        sub_vals,sub_locs = runSubBurstNnf(search_frames, 1, nblocks, k=1,
                                           blockLabels=search_blocks,
                                           img_shape=img_shape,valMean=valMean)
        sub_vals = sub_vals / center_crop(nuniuqes,ishape)[...,None]
        sub_vals = torch.abs(sub_vals - valMean)
        sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')
        assert torch.all(sub_locs[0] == 0).item() is True, "all zero @ ref."

        # -- 3.) update vals and locs --
        vals,locs = update_state_outliers(vals,locs,sub_vals,
                                          sub_locs,names,False)
        # print(locs[:,0,16,16,0,:])
        max_displ = torch.abs(locs).max().item()
        # assert max_displ <= nbHalf, "displacement must remain contained!"

        # -- 4.) rewarp bursts --
        pad_locs = padLocs(locs,nbHalf,'extend')
        nframes,nimages,hP,wP,k,two = pad_locs.shape
        warped_noisy_old = warped_noisy.clone()
        warped_noisy = warp_burst_from_locs(tnoisy,pad_locs,nblocks,psize)[0]

        delta = torch.sum(torch.abs(prev_locs - locs)).item()
        # print("Delta: ",delta)

    # -------------------------------
    #
    # --    finalizing outputs     --
    #
    # -------------------------------
    # print("Counts: ",counts)

    # -- get the output image from tiled image --
    # warped_noisy = center_crop(warped_noisy,ishape)
    # warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    
    # -- convert "locs" to "flow" --
    if to_flow:
        locs = locs2flow(locs)

    # -- reformat for experiment api --
    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs
