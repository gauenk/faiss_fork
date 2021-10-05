"""
Belief Propogation Search

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

from .utils import create_search_ranges,warp_burst_from_pix,warp_burst_from_locs,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs
from .merge_search_ranges_numba import merge_search_ranges

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def runBpSearch(noisy, clean, patchsize, nblocks, k = 1,
                nparticles = 1, niters = 10,
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
    # print("pix.shape ",pix.shape)
    pix = pix.to(device)
    # print("l2_locs.shape ",l2_locs.shape)
    locs = torch.zeros_like(pix2locs(pix))
    l2_locs = locs

    # # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)
    search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)
    # names = torch.ones((nimages,nframes,1,h,w)).to(pix.device)
    # for t in range(nframes): names[:,t] = t
    # merged_search_ranges = merge_search_ranges(pix,names,search_ranges,
    #                                            nblocks,pixAreLocs=False,
    #                                            drift=False)

    # -- 3.) warp burst to top location --
    wnoisy = padAndTileBatch(noisy,patchsize,nblocks)
    wclean = padAndTileBatch(clean,patchsize,nblocks)
    pixPad = (wnoisy.shape[-1] - noisy.shape[-1])//2
    ppix = padLocs(pix+pixPad,pixPad)
    plocs = padLocs(locs,pixPad)

    print("plocs.shape ",plocs.shape)
    print("wnoisy.shape ",wnoisy.shape)
    warped_noisy = warp_burst_from_locs(wnoisy,plocs,1,pisize)
    warped_clean = warp_burst_from_locs(wclean,plocs,1,pisize)
    # warped_noisy = warp_burst_from_pix(wnoisy,ppix,nblocks)
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

    # vals,e_locs = evalAtLocs(noisy, sub_locs_rs_pad,
    #                          patchsize, nblocks,
    #                          img_shape=None)
    vals,e_locs = evalAtLocs(wnoisy,
                             sub_locs_rs, 1,
                             nblocks,
                             img_shape=img_shape)
    # vals,e_locs = evalAtLocs(wnoisy,
    #                          sub_locs_rs_nb_pad, 1,
    #                          nblocks,
    #                          img_shape=img_shape)
    # vals,e_locs = runSubBurstNnf(noisy,patchsize,
    #                              nblocks,k=1, # patchsize=1 since tiled
    #                              blockLabels=None)
    # print("[SubBurst]: Vals (5,5): ",vals[:,5,5])

    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    sub_locs_rs_pad = padLocs(sub_locs_rs,psHalf,mode='extend')
    # sub_locs_rs_nb_pad = padLocs(sub_locs_rs,padSize,mode='extend')
    # print("sub_locs_rs_nb_pad.shape ",sub_locs_rs_nb_pad.shape)
    psHalf = patchsize//2
    nbHalf = nblocks//2
    padSize = psHalf


    clK = [4,]*niters#niters # -- scheduler for clustering --
    # search_blocks = compute_search_blocks(search_ranges,3) # -- matches clK --
    # K = nframes

    exp_locs = locs
    for i in range(niters):

        # -- 1.) cluster each pixel across time --
        K = clK[i]
        # print("wnoisy.shape: ",wnoisy.shape)
        # print("[start] compute_temporal_cluster")
        # cc_wnoisy = center_crop(warped_noisy,pshape)
        names,means,weights,mask = compute_temporal_cluster(warped_noisy,K)
        wmeans = means * weights
        wmeans = wmeans[0] # nparticles == 1
        wmeans = wmeans.contiguous()
        cc_names = center_crop(names,ishape)
        # print("[end] compute_temporal_cluster")
        # print(means.shape)

        # print("search_ranges.shape ",search_ranges.shape)
        # print("msr.shape ",msr.shape)
        # locs = split_locs(best_locs,offsets,cc_names,msr,nblocks)
        # locs_frames2groups(locs,cc_names,search_ranges,nblocks)
        # print(glocs)
        # print(glocs.shape)
        # exit()

        # # -- 2.) denoise each group (e.g. use averages) --
        # dclusters = denoise_clustered_noisy(wnoisy,clusters,ave_denoiser)
        # -- skip for now --

        # -- 3.) create combinatorial search blocks from search ranges  --
        # print("[start] merge_search_ranges")
        refG = nframes//2
        # we pad BOTH but patchsize == 1 so it's only nbHalf that is included.
        pad_locs = padLocs(locs,nbHalf,'extend') # since patchsize is 1 here.
        # pad_locs = padLocs(locs,nblocks//2) # since patchsize is 1 here.
        # print("pad_search_ranges.shape ",pad_search_ranges.shape)
        # print("locs.shape ",locs.shape)
        # print("locs[:,:,5,6]: ",locs[:,:,5,6])
        # print("pad_locs.shape ",pad_locs.shape)
        # print("pad_locs[:,:,5,6]: ",pad_locs[:,:,5+nblocks//2,6+nblocks//2])
        # print("pad_locs.shape ",pad_locs.shape)
        # print("names.shape ",names.shape)
        merged_search_ranges,offsets = merge_search_ranges(pad_locs,names,
                                                           pad_search_ranges,
                                                           nblocks,pixAreLocs=True)
        # print(search_blocks[:,0,16+nbHalf,16+nbHalf])
        # print("merged_search_ranges.shape ",merged_search_ranges.shape)
        # # msr_mid = merged_search_ranges[:,0,5+nbHalf,16+nbHalf]
        # msr_mid = merged_search_ranges[:,0,5+nbHalf,6+nbHalf]

        # print(f"[PLocs]: (5,6): ",pad_locs[:,:,5+nbHalf,6+nbHalf])
        # print("names: (5,6): ",names[:,:,:,5+nbHalf,6+nbHalf])
        # ngroups = msr_mid.shape[1]
        # for gID in range(ngroups):
        #     msr_max0 = msr_mid[:,gID,0].max().item()
        #     msr_min0 = msr_mid[:,gID,0].min().item()

        #     msr_max1 = msr_mid[:,gID,1].max().item()
        #     msr_min1 = msr_mid[:,gID,1].min().item()

        #     print(f"[MSR {gID}] (5,6) 0: ",msr_max0,msr_min0,msr_mid.shape)
        #     print(f"[MSR {gID}] (5,6) 1: ",msr_max1,msr_min1,msr_mid.shape)

        # print(msr_mid[:,0].max(),msr_mid[:,0].min())
        # print(pad_locs[:,0,5+nbHalf,5+nbHalf])
        # print(names[:,:,:,5+nbHalf,5+nbHalf])
        # sb_mid = search_blocks[:,0,5+nbHalf,5+nbHalf]
        # print(sb_mid.max(),sb_mid.min())
        # exit()

        search_blocks = compute_search_blocks(merged_search_ranges,refG)

        search_blocks = rearrange(search_blocks,'l i h w t two -> l i t two h w')
        search_blocks = center_crop(search_blocks,img_shape[1:])
        search_blocks = rearrange(search_blocks,'l i t two h w -> l i h w t two')
        search_blocks = search_blocks.contiguous()


        # print("search_blocks.shape ",search_blocks.shape)

        # merged_search_ranges = merge_search_ranges(pix,cc_names,search_ranges,
        #                                            nblocks,pixAreLocs=False)
        # search_blocks = compute_search_blocks(merged_search_ranges,refG)
        # print("search_blocks.shape: ",search_blocks.shape)
        # print("(5,6)")
        # sb_mid = search_blocks[:,0,5,6,:,:]
        # print(sb_mid[:,0].max(),sb_mid[:,0].min())
        # print("0: ",sb_mid[:,0,0].max(),sb_mid[:,0,0].min())
        # print("1: ",sb_mid[:,0,1].max(),sb_mid[:,0,1].min())

        # print("(5,5)")
        # sb_mid = search_blocks[:,0,5,5,:,:]
        # print(sb_mid[:,0].max(),sb_mid[:,0].min())
        # print("0: ",sb_mid[:,0,0].max(),sb_mid[:,0,0].min())
        # print("1: ",sb_mid[:,0,1].max(),sb_mid[:,0,1].min())


        # print("[end] merge_search_ranges")
        # print("Extrema of Search Blocks: ",search_blocks.max(),search_blocks.min())
        
        # -- 4.) exh srch over a clusters --
        in_burst = wmeans
        # in_burst = warped_burst[0]
        # in_burst = wburst
        # print("in_burst.shape: ",in_burst.shape)
        sub_vals,sub_locs = runSubBurstNnf(in_burst,1,
                                           nblocks,k=1, # patchsize=1 since tiled
                                           mask=mask,
                                           blockLabels=search_blocks,
                                           img_shape = img_shape)
        # print("ASDF")

        # print("vals: ",vals.shape)
        # print("locs: ",locs.shape)
        # print("sub_vals: ",sub_vals.shape)
        # print("sub_locs: ",sub_locs.shape)
        sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')
        # print("[post] sub_locs: ",sub_locs.shape)
        # hIdx,wIdx = 5,6
        # print(vals[:,hIdx,wIdx])
        # print(sub_vals[:,hIdx,wIdx])
        # vals = sub_vals
        # locs = sub_locs + locs
        # print("iter: ",i)
        # print(locs[:,:,5,5])
        # print(sub_locs[:,:,5,5])
        # print(names[0,:,0,6,6])
        
        # print("names: ",names.shape)
        # print(vals[...,32,32,0])
        # sub_vals,sub_locs = runWeightedBurstNnf(means,weights,patchsize,nblocks,
        #                                         blockLabels=search_blocks)

        # -- 5.) update vals and locs --
        # overwrite = i == 0

        # print("vals.shape ",vals.shape)
        # print("locs.shape ",locs.shape)
        # print("names.shape ",names.shape)
        # names = names[0] # nparticles == 1
        cc_names = center_crop(names[...,0,:,:],img_shape[1:])
        # diff = torch.abs(vals[:,0,1] - 3.6135).item()
        # print("[vals[0,1] == 3.6135]?",vals[:,0,1],diff)
        # diff = torch.abs(sub_vals[:,0,1] - 3.6135).item()
        # print("[sub_vals[0,1] == 3.6135]?",sub_vals[:,0,1],diff)
        # print("search_blocks[0,1]: ",search_blocks[:,0,0+1,1+1,:])
        # print("names[0,1]: ",names[0,:,0+1,1+1])
        # print("search_blocks.shape ",search_blocks.shape)

        # print("(5,5)")
        # print("vals",vals[0,5,5,0].item(),vals.shape)
        # print("sub_vals",sub_vals[0,5,5,0].item(),sub_vals.shape)
        # print("pre locs ",locs[:,:,5,5])
        # print("sub locs ",sub_locs[:,:,5,5])
        # print(cc_names[0,:,5,5])

        # print("-"*30)
        # print("(5,6)")
        # print("vals",vals[0,5,6,0].item(),vals.shape)
        # print("sub_vals",sub_vals[0,5,6,0].item(),sub_vals.shape)
        # print("pre locs ",locs[:,:,5,6],locs.shape)
        # print("sub locs ",sub_locs[:,:,5,6],sub_locs.shape)
        # print(cc_names[0,:,5,6])


        # e_vals,e_locs = evalAtLocs(warped_noisy[0],
        #                            sub_locs, 1,
        #                            nblocks,
        #                            img_shape=img_shape)
        # print("warped_noisy @ SubLocs")
        # print("[Eval At SubLocs]: (5,6) ",e_vals[:,5,6])
        # assert torch.sum(torch.abs(e_vals - vals)) < 1e-5,"Equal vals."


        # offsets = rearrange(offsets,'i p h w t two -> i p t two h w')
        # cc_offsets = center_crop(offsets,ishape)
        # cc_offsets = rearrange(cc_offsets,'i p t two h w -> i p h w t two')
        # print("cc_names.shape ",cc_names.shape)
        # vals = sub_vals
        # locs = sub_locs
        # print("sub_locs.shape: ",sub_locs.shape)
        # print("offsets.shape: ",offsets.shape)
        # print("cc_offsets.shape: ",cc_offsets.shape)
        # cc_offsets = rearrange(cc_offsets,'i p h w t two -> t i h w p two')
        # print("[post] cc_offsets.shape: ",cc_offsets.shape)
        # print("cc_offsets[...,5,6,...]: ",cc_offsets[:,0,5,6,0])
        # print("pre ",sub_locs[:,:,5,6])
        # # sub_locs -= cc_offsets
        # print("post ",sub_locs[:,:,5,6])
        # # exit()
        
        vals,locs,exp_locs = update_state(vals,locs,sub_vals,sub_locs,cc_names,False)

        # print("[post update] (5,6)")
        # print("vals",vals[0,5,6,0].item(),vals.shape)
        # print("locs ",locs[:,:,5,6],locs.shape)
        # print(cc_names[0,:,5,6])

        # print("POST")
        # print("(16,16): \n",locs[:,:,16,16])
        # print("(5,6): \n",locs[:,:,5,6])
        # for i in range(h):
        #     for j in range(h):
        #         print("post ",locs[:,:,i,j],(i,j))
        
        max_displ = np.max([locs.max().item(),np.abs(locs.min().item())])
        # print(max_displ)
        assert max_displ <= nblocks//2, "displacement must remain contained!"


        # -- 6.) rewarp bursts --
        pad_locs = padLocs(locs,nbHalf,'extend')
        # print("locs.shape: ",pad_locs.shape)
        nframes,nimages,hP,wP,k,two = pad_locs.shape
        psize = edict({'h':hP,'w':wP})
        # old_warped_burst = warped_burst

        # pad_locs_zero = torch.zeros_like(pad_locs)
        # warped_noisy = warp_burst_from_locs(wnoisy,pad_locs,nblocks,psize)
        p_exp_locs = padLocs(exp_locs,nbHalf,'extend')
        # print("pad_locs.shape ",pad_locs.shape)
        # print("p_exp_locs.shape ",p_exp_locs.shape)
        warped_noisy = warp_burst_from_locs(warped_noisy[0],p_exp_locs,nblocks,psize)
        warped_clean = warp_burst_from_locs(warped_clean[0],p_exp_locs,nblocks,psize)

        # exit()
        # print("[post update]: (16,17) ",vals[:,16,17])
        # print("[post update]: (17,16) ",vals[:,17,16])

        # e_vals,e_locs = evalAtLocs(wnoisy,
        #                          locs, 1,
        #                          nblocks,
        #                          img_shape=img_shape)

        # print("wnoisy @ loc")
        # print("[Eval At Locs]: (5,6) ",e_vals[:,5,6])
        # print("[Eval At Locs]: (17,16) ",vals[:,17,16])
        # assert torch.sum(torch.abs(e_vals - vals)) < 1e-5,"Equal vals."

        # e_vals,e_locs = evalAtLocs(warped_noisy[0],
        #                            torch.zeros_like(locs), 1,
        #                            nblocks,
        #                            img_shape=img_shape)

        # print("warped_noisy @ zeros")
        # print("[Eval At Locs]: (5,6) ",e_vals[:,5,6])
        # # print("[Eval At Locs]: (17,16) ",vals[:,17,16])
        # assert torch.sum(torch.abs(e_vals - vals)) < 1e-5,"Equal vals."


        # warped_noisy_v2 = warp_burst_from_locs(wnoisy,pad_locs,nblocks,psize)

        # vals,e_locs = evalAtLocs(warped_noisy_v2[0],
        #                          torch.zeros_like(locs), 1,
        #                          nblocks,
        #                          img_shape=img_shape)


        # print("warped at zoer")
        # print("[Eval At Locs]: (16,17) ",vals[:,16,17])
        # print("[Eval At Locs]: (17,16) ",vals[:,17,16])

        # print(torch.sum(torch.abs(wmeans - warped_noisy[0])))
        # vals,e_locs = evalAtLocs(wmeans,#warped_noisy[0],
        #                          sub_locs, 1,
        #                          nblocks,
        #                          img_shape=img_shape)
        # print("wmean")
        # print("[Eval At Locs]: (16,17) ",vals[:,16,17])
        # print("[Eval At Locs]: (17,16) ",vals[:,17,16])

        # vals,e_locs = evalAtLocs(old_warped_noisy[0],
        #                          sub_locs, 1,
        #                          nblocks,
        #                          img_shape=img_shape)
        # print("old_warped_noisy[0]")
        # print("[Eval At Locs]: (16,17) ",vals[:,16,17])
        # print("[Eval At Locs]: (17,16) ",vals[:,17,16])

        # vals,e_locs = evalAtLocs(warped_noisy[0],
        #                          torch.zeros_like(sub_locs), 1,
        #                          nblocks,
        #                          img_shape=img_shape)
        # print("warped_noisy[0] @ zeros")
        # print("[Eval At Locs]: (16,17) ",vals[:,16,17])
        # print("[Eval At Locs]: (17,16) ",vals[:,17,16])


        # exit()

    warped_noisy = center_crop(warped_noisy[0],ishape)
    warped_clean = center_crop(warped_clean[0],ishape)
    # print(warped_noisy.shape)
    # print(warped_clean.shape)
    # flows.bp_est = rearrange(flow_gt,'i t h w two -> i (h w) t two')
    # warped_clean = rearrange(warped_noisy,'t i f h w -> i (h w) t two')
    def index_along_ftrs(warped_tiled,ps,c):
        assert warped_tiled.shape[-3] == ps**2 * c, "ensure eq dims"
        ps2 = patchsize**2
        psMid = ps2//2
        fIdx = torch.arange(psMid,ps2*c,ps2)
        warped = warped_tiled[...,fIdx,:,:]
        return warped
    warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    warped_clean = index_along_ftrs(warped_clean,patchsize,c)
    
    if to_flow:
        locs = locs2flow(locs)

    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs,warped_noisy,warped_clean
