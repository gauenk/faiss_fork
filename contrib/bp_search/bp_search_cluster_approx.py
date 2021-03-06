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
from nnf_share import padBurst,getBlockLabels,tileBurst,padAndTileBatch,padLocs,locs2flow,flow2locs
# from bnnf_utils import runBurstNnf
from sub_burst import runBurstNnf as runSubBurstNnf
from sub_burst import evalAtLocs
# from wnnf_utils import runWeightedBurstNnf
from easydict import EasyDict as edict

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import save_image

from .utils import create_search_ranges,warp_burst_from_pix,warp_burst_from_locs,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs,index_along_ftrs,flow_to_groups,cluster_frames_by_groups,update_state_locs,compute_mode,grouped_flow,smooth_locs,clip_loc_boarders
from .merge_search_ranges_numba import merge_search_ranges
from .approx_exh import runBpSearchApproxExh

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def runBpSearchClusterApprox(noisy, clean, patchsize, nblocks, k = 1,
                             nparticles = 1, niters = 3,
                             valMean = 0.,std=None,
                             l2_nblocks = None, l2_valMean=0.,
                             blockLabels=None, ref=None,
                             to_flow=False, fmt=False, gt_info=None):

    nframes,nimages,c,h,w = noisy.shape
    if l2_nblocks is None: l2_nblocks = nblocks
    assert nparticles == 1, "Only one particle currently supported."
    ngroups = -1
    if not(gt_info is None):
        flow = gt_info['flow']
        groups,ngroups = flow_to_groups(flow)
        groups = groups.to(noisy.device,non_blocking=True) # (t i h w)
        locs_gt = flow2locs(flow)
        locs_gt = rearrange(locs_gt,'i t h w two -> 1 i h w t two')
    else:
        groups = torch.zeros((nframes,nimages,h,w)).type(torch.long)
    print("ngroups: ",ngroups)

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = noisy.device
    img_shape = [c,h,w]
    pad = 2*(nblocks//2)
    ishape = [h,w]
    isize = edict({'h':h,'w':w})
    pisize = edict({'h':h+pad,'w':w+pad})
    l2_pad = 2*(l2_nblocks//2)
    l2_pisize = edict({'h':h+l2_pad,'w':w+l2_pad})
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
    # locs[1,...,0] = 1
    # locs[1,...,1] = -1
    # locs[6,...,0] = 0
    # locs[6,...,1] = -1
    # locs = torch.zeros_like(locs)
    # locs = smooth_locs(locs,nclusters=3)
    # locs = clip_loc_boarders(locs,patchsize,l2_nblocks,nblocks)
    l2_locs = locs
    # exit()

    # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)
    search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)

    # -- 3.) warp burst to top location --
    wnoisy = padAndTileBatch(noisy,patchsize,l2_nblocks)
    wclean = padAndTileBatch(clean,patchsize,l2_nblocks)
    pixPad = (wnoisy.shape[-1] - noisy.shape[-1])//2
    ppix = padLocs(pix+pixPad,pixPad)
    plocs = padLocs(locs,pixPad)
    # print(wnoisy.shape)
    # print(noisy.shape)
    # print(plocs.shape)

    # print("pre warp!")
    warped_noisy = warp_burst_from_locs(wnoisy,plocs,l2_pisize)
    warped_clean = warp_burst_from_locs(wclean,plocs,l2_pisize)
    img_shape[0] = wnoisy.shape[-3]
    # print("post warp!")
    # print(warped_noisy.shape)
    
    hP,wP = h + 2*pixPad,w + 2*pixPad
    pad_search_ranges = create_search_ranges(nblocks,hP,wP,nframes)
    pad_search_ranges = torch.LongTensor(pad_search_ranges[:,None]).to(device)

    save_image("noisy.png",noisy)

    # -------------------------------
    #
    #   Correctly set values using  
    #   Sub-Ave instead of L2
    #
    # -------------------------------
    sub_vals = l2_vals
    sub_locs_rs = l2_locs
    psHalf = patchsize//2
    # print("pre eval.")
    # print(wnoisy.shape)
    # print(sub_locs_rs.shape)
    # print(img_shape)
    vals,e_locs = evalAtLocs(wnoisy,sub_locs_rs, 1,
                             l2_nblocks,img_shape=img_shape)
    # print("post eval.")

    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------


    # -- crop warped frame as init location --
    warped_noisy = center_crop(warped_noisy,pshape)
    wnoisy = warped_noisy[0].clone()
    locs = torch.zeros_like(locs)

    # -- params --
    psHalf = patchsize//2
    nbHalf = nblocks//2
    padSize = psHalf
    pixPad = (wnoisy.shape[-1] - img_shape[-1])//2

    # -- loops stuff --
    clK = [3,]*niters#niters # -- scheduler for clustering --
    # search_blocks = compute_search_blocks(search_ranges,3) # -- matches clK --
    # K = nframes
    # modes = compute_mode(std,patchsize**2,groups)
    # mode = torch.mean(modes*ngroups).item()
    mode = 0.
    refGroup = groups[nframes//2,0,16,16].item()
    mask = torch.ones_like(groups).type(torch.bool).contiguous()
    groups_known = True

    # print("-"*30)
    # print(locs[:,0,16,16,0])

    exp_locs = locs
    clK = [5,]
    niters = len(clK)
    for i in range(niters):

        # print("pre cluster!")
        # -- 1.) cluster each pixel across time --
        if False and groups_known:
            # -- known groups --
            pad = nbHalf
            pad_groups = th_pad(groups,(pad,)*4,padding_mode='reflect')
            wmeans = cluster_frames_by_groups(warped_noisy[0],pad_groups,ngroups)
            names = pad_groups[None,:]
            ud_names = groups
            refG = refGroup
        else:
            # -- unknown groups --
            K = clK[i]
            names,means,weights,mask = compute_temporal_cluster(warped_noisy,K)
            wmeans = means /weights #* weights/nframes
            wmeans = wmeans[0] # nparticles == 1
            wmeans = wmeans.contiguous()
            cc_names = center_crop(names,ishape)
            ud_names = center_crop(rearrange(names,'1 t i h w -> t i h w'),ishape)
            ngroups = names.max().item() + 1
            refG = names[0,nframes//2,0,16,16].item()
            refGroup = refG
        # print("names.shape: ",names.shape)
        # print("ud_names.shape: ",ud_names.shape)
        # print("post cluster!")
        
        # -- 2.) create combinatorial search blocks from search ranges  --
        pad_locs = padLocs(locs,nbHalf,'extend')
        merged_search_ranges,offsets = merge_search_ranges(pad_locs,names,
                                                           pad_search_ranges,
                                                           nblocks,pixAreLocs=True,
                                                           drift=True)
        # print("merged_search_ranges.shape: ",merged_search_ranges.shape)
        # merged_search_ranges = torch.flip(merged_search_ranges,dims=(-1,))
        # search_blocks = compute_search_blocks(merged_search_ranges,refG)
        msr = merged_search_ranges[:,0,16,16,:,:]
        msr = rearrange(msr,'l t two -> t l two')
        # print(msr,refG)

        # assert torch.all(search_blocks[...,refGroup,:]==0).item() is True,"no search."
        # print(names[0,:,0,18,18])
        # print(pad_locs[:,0,16,16,0,:])
        # print_msr = merged_search_ranges[:,0,16,16,:,:]
        # print("print_msr.shape: ",print_msr.shape)
        # print_msr = rearrange(print_msr,'r t two -> t r two')
        # print(print_msr)
        # exit()

        # -- 3.) apprx exh srch over a clusters --
        # print("\n\n\n\n Approx. \n\n\n\n")
        sub_vals,sub_locs = runBpSearchApproxExh(wmeans,1,nblocks,k=1,
                                                 valMean=0.,#mode,
                                                 ref=refGroup,
                                                 search_ranges=None,#merged_search_ranges,
                                                 blockLabels=None,#search_blocks,
                                                 niters=3,
                                                 img_shape = img_shape)
        # print("post approx.")
        # print("search_blocks.shape: ",search_blocks.shape)
        # print("wnoisy.shape: ",wnoisy.shape)
        # sub_vals,sub_locs = runSubBurstNnf(wmeans,1,nblocks,k=1,
        #                                    valMean=0.,#mode,
        #                                    blockLabels=search_blocks,
        #                                    img_shape = img_shape)
        # sub_vals,sub_locs = runSubBurstNnf(noisy,patchsize,nblocks,k=1)
        # sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')
        # print("sub_locs.shape: ",sub_locs.shape)
        # print("locs.shape: ",locs.shape)
        # print("[a]: ",sub_vals[0,16,16,0].item())
        # print("[a]:\n",sub_locs[:,0,16,16,0,:])

        # -- formatting --
        sub_locs = sub_locs.type(torch.long)
        # print("Complete.")
        esub_locs = update_state_locs(locs,sub_locs,ud_names)
        locs = esub_locs

        # print("pre warped.")
        plocs = padLocs(locs,pixPad)
        # print("wnoisy.shape: ",wnoisy.shape)
        # print("plocs.shape: ",plocs.shape)
        # print("locs.shape: ",locs.shape)
        # print("pixPad: ",pixPad)
        # print(pisize)
        warped_noisy = warp_burst_from_locs(wnoisy,plocs,pisize)
        # print("post warped.")

        # print("-"*30)
        # print(locs[:,0,16,16,0,:])

        # locs_gt = flow2locs(gt_info['flow'])
        # locs_gt = rearrange(locs_gt,'i t h w two -> t i h w 1 two')
        # print("locs_gt")
        # print(locs_gt[:,0,16,16,0,:])

        # eqs = torch.all(torch.all(locs == locs_gt,dim=-1),dim=0)
        # perc_eq = torch.mean(eqs.type(torch.float)).item()
        # print("perc_eq: ",perc_eq)
        # print(torch.where(eqs == 0))
        # print("eqs.shape: ",eqs.shape)
        # eq_img = rearrange(eqs.type(torch.float),'i h w 1 -> i 1 h w')
        # print("eq_img.shape: ",eq_img.shape)
        # save_image(eq_img,"eq_image.png")
        # print("search_blocks.shape: ",search_blocks.shape)

        # locs_gt_srch = rearrange(locs_gt,'t i h w 1 two -> 1 i h w t two')
        # sub_vals,sub_locs = runSubBurstNnf(wnoisy,1,nblocks,k=1,
        #                                    valMean=0.,
        #                                    blockLabels=locs_gt_srch,
        #                                    img_shape = img_shape)
        # sample_optimal = sub_vals[0,16,16,0].item()
        # print("Sample Optimal Value v.s. Computed Mode")
        # print(sample_optimal,mode)

        # -- 5.) re-compute vals from proposed locs --
        # ud_names = groups
        # print("TYPES")
        # print(locs.type())
        # print(sub_locs.type())
        # print(ud_names.type())
        # print(locs.shape)
        # print(sub_locs.shape)
        # print(ud_names.shape)
        # pad_locs = padLocs(locs,nbHalf,'extend')
        # sub_locs = sub_locs.type(torch.long)
        # print("locs.shape: ",locs.shape)
        # print("sub_locs.shape: ",sub_locs.shape)
        # print("ud_names.shape: ",ud_names.shape)
        # search_locs = update_state_locs(locs,sub_locs,ud_names)
        # print("search_locs.shape: ",search_locs.shape)
        # print("search_locs[:,0,16,16,0,:]")
        # print(search_locs[:,0,16,16,0,:])


        # # print("search_blocks.shape: ",search_blocks.shape)
        # # print("locs.shape: ",locs.shape)
        # # print("sub_locs.shape: ",sub_locs.shape)
        # print("search_locs.shape: ",search_locs.shape)
        # search_locs = rearrange(search_locs,'t i h w k two -> k i h w t two')
        # search_locs = search_locs.contiguous()
        # # print("search_locs.shape: ",search_locs.shape)
        # prop_vals,prop_locs = runSubBurstNnf(wnoisy,1,nblocks,k=1,
        #                                    blockLabels=search_locs,
        #                                    img_shape=img_shape)
        # prop_locs = rearrange(prop_locs,'i t h w k two -> t i h w k two')

        # # -- 6.) update vals and locs --
        # cc_names = center_crop(names[...,0,:,:],img_shape[1:])
        # vals,locs,exp_locs = update_state(vals,locs,prop_vals,prop_locs,cc_names,False)
        # max_displ = np.max([locs.max().item(),np.abs(locs.min().item())])
        # assert max_displ <= nblocks//2, "displacement must remain contained!"


        # # -- 7.) rewarp bursts --
        # pad_locs = padLocs(locs,nbHalf,'extend')
        # nframes,nimages,hP,wP,k,two = pad_locs.shape
        # psize = edict({'h':hP,'w':wP})
        # # p_exp_locs = padLocs(exp_locs,nbHalf,'extend')
        # warped_noisy = warp_burst_from_locs(wnoisy,pad_locs,nblocks,psize)
        # warped_clean = warp_burst_from_locs(wclean,pad_locs,nblocks,psize)

    warped_noisy = center_crop(wnoisy,ishape)
    warped_clean = center_crop(wclean,ishape)

    warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    warped_clean = index_along_ftrs(warped_clean,patchsize,c)
    
    # -- add back the l2_locs --
    locs += l2_locs

    if to_flow:
        locs = locs2flow(locs)

    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs,warped_noisy#,warped_clean
