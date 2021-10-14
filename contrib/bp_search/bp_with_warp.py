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

from .utils import create_search_ranges,warp_burst_from_pix,warp_burst_from_locs,compute_temporal_cluster,update_state,locs_frames2groups,compute_search_blocks,pix2locs,index_along_ftrs,flow_to_groups,cluster_frames_by_groups,update_state_locs,compute_mode,grouped_flow
from .merge_search_ranges_numba import merge_search_ranges

center_crop = torchvision.transforms.functional.center_crop
resize = torchvision.transforms.functional.resize
th_pad = torchvision.transforms.functional.pad

def runBpSearch(noisy, clean, patchsize, nblocks, k = 1,
                nparticles = 1, niters = 1,
                valMean = 0.,std=None,
                l2_nblocks = None, l2_valMean=0.,
                blockLabels=None, ref=None,
                to_flow=False, fmt=False, gt_info=None,
                img_shape = None):

    if l2_nblocks is None: l2_nblocks = nblocks
    assert nparticles == 1, "Only one particle currently supported."
    ngroups = -1
    if not(gt_info is None):
        flow = gt_info['flow']
        groups,ngroups = flow_to_groups(flow)
        groups = groups.to(noisy.device,non_blocking=True)
        locs_gt = flow2locs(flow)
        locs_gt = rearrange(locs_gt,'i t h w two -> 1 i h w t two')

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = noisy.device
    nframes,nimages,c,h,w = noisy.shape
    if img_shape is None: img_shape = [c,h,w]
    c,h,w = img_shape
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
    locs = torch.zeros_like(pix2locs(pix))
    l2_locs = locs

    # -- 2.) create local search radius from topK locs --
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
    vals,e_locs = evalAtLocs(wnoisy,sub_locs_rs, 1,
                             nblocks,img_shape=img_shape)

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
    modes = compute_mode(std,patchsize**2,groups)
    mode = torch.mean(modes*ngroups).item()
    # print("groups.shape: ",groups.shape)
    refGroup = groups[nframes//2,0,16,16].item()
    mask = torch.ones_like(groups).type(torch.bool).contiguous()

    exp_locs = locs
    for i in range(niters):

        # print("warped_noisy.shape: ",warped_noisy.shape)
        # -- 1.) cluster each pixel across time --
        # K = clK[i]
        # names,means,weights,mask = compute_temporal_cluster(warped_noisy,K)
        # wmeans = means * weights/nframes
        # wmeans = wmeans[0] # nparticles == 1
        # wmeans = wmeans.contiguous()
        # cc_names = center_crop(names,ishape)
        # print("[1] wmeans.shape: ",wmeans.shape)
        # print("[1] names.shape: ",names.shape)

        # -- 1.) [version 2] known groups --
        pad = int(nblocks//2)
        pad_groups = th_pad(groups,(pad,)*4,padding_mode='constant')
        wmeans = cluster_frames_by_groups(warped_noisy[0],pad_groups,ngroups)
        names = pad_groups[None,:]
        ud_names = groups
        # print("[2] wmeans.shape: ",wmeans.shape)
        # print("[2] names.shape: ",names.shape)
        # print(pad_groups[:,0,16,16])
        # print(wmeans[:,0,0,16,16])
        # print(warped_noisy[0,:,0,0,16,16])
        # print(wmeans[:,0,0,16,16].shape)
        # print(warped_noisy[0,:,0,0,16,16].shape)

        # exit()
        

        # # -- 2.) denoise each group (e.g. use averages) --
        # dclusters = denoise_clustered_noisy(wnoisy,clusters,ave_denoiser)
        # -- skip for now --

        # -- 3.i) create combinatorial search blocks from search ranges  --
        refG = nframes//2
        pad_locs = padLocs(locs,nbHalf,'extend') # since patchsize is 1 here.
        merged_search_ranges,offsets = merge_search_ranges(pad_locs,names,
                                                           pad_search_ranges,
                                                           nblocks,pixAreLocs=True)
        # print("merged_search_ranges[...]:\n",merged_search_ranges[:,0,16,16,:,:])
        # print("merged_search_ranges[...]:\n",merged_search_ranges[:,0,16,16,4,:])
        # print("merged_search_ranges.shape: ",merged_search_ranges.shape)

        # -- 3.ii) search space --
        refG = refGroup
        search_blocks = compute_search_blocks(merged_search_ranges,refG)
        # print(search_blocks[:,0,16,16,4,:])
        # print(search_blocks[:,0,16,16,4,:].shape)
        # for i in range(search_blocks.shape[0]):
        #     print(search_blocks[i,0,16,16,4,:])
        search_blocks = rearrange(search_blocks,'l i h w t two -> l i t two h w')
        search_blocks = center_crop(search_blocks,img_shape[1:])
        search_blocks = rearrange(search_blocks,'l i t two h w -> l i h w t two')
        search_blocks = search_blocks.contiguous()

        # nbHalf = int(nblocks//2)
        # sblocks = search_blocks[:,0,20,20,:,:].type(torch.long)
        # square = torch.zeros(ngroups,nblocks,nblocks)
        # for i in range(sblocks.shape[0]):
        #     for t in range(ngroups):
        #         a = sblocks[i,t,0].item()+nbHalf
        #         b = sblocks[i,t,1].item()+nbHalf
        #         square[t,a,b] += 1
        # print(square)
        # exit()

        
        # -- 4.) exh srch over a clusters --
        print("search_blocks.shape: ",search_blocks.shape)
        sub_vals,sub_locs = runSubBurstNnf(wmeans,1,nblocks,k=1,# patchsize=1 since tiled
                                           mask=mask,valMean=mode,
                                           blockLabels=search_blocks,
                                           img_shape = img_shape)
        sub_locs = rearrange(sub_locs,'i t h w k two -> t i h w k two')
        sub_locs = sub_locs.type(torch.long)

        # print("sub_vals.shape: ",sub_vals.shape)
        # print("sub_locs[:,0,16,16,0,:]")
        # print(sub_locs[:,0,16,16,0,:])
        # print("at searched: ",sub_vals[0,16,16,0].item())

        # print("locs_gt.shape: ",locs_gt.shape)
        # pad_locs_gt = padLocs(locs_gt,nbHalf,'extend')
        # print("pad_locs_gt.shape: ",pad_locs_gt.shape)
        # print("sub_locs.shape: ",sub_locs.shape)
        esub_locs = update_state_locs(locs,sub_locs,ud_names)
        locs = esub_locs

        # pad_esub_locs = padLocs(esub_locs,nbHalf,'extend')
        # print("pad_esub_locs.shape: ",pad_esub_locs.shape)
        # # pad_locs_gt = rearrange(pad_locs_gt,'i t h w two -> 1 i h w t two')
        # sub_vals,sub_locs = runSubBurstNnf(wnoisy,1,nblocks,k=1,
        #                                    # patchsize=1 since tiled
        #                                    mask=mask,valMean=valMean,
        #                                    blockLabels=pad_esub_locs,
        #                                    img_shape = img_shape)

        # print("sub_vals.shape: ",sub_vals.shape)
        # print("at searched optimal: ",sub_vals[0,16,16,0])

        # sub_vals,sub_locs = runSubBurstNnf(wnoisy,1,
        #                                    nblocks,k=1, # patchsize=1 since tiled
        #                                    mask=mask,valMean=valMean,
        #                                    blockLabels=pad_locs_gt,
        #                                    img_shape = img_shape)

        # print("sub_vals.shape: ",sub_vals.shape)
        # print("at optimal: ",sub_vals[0,16,16,0].item())

        # gflow = grouped_flow(flow,groups)
        # glocs = flow2locs(gflow)
        # glocs = rearrange(glocs,'i t h w two -> 1 i h w t two')
        # print("glocs.shape: ",glocs.shape)
        # pad_glocs = padLocs(glocs,nbHalf,'extend')
        # sub_vals,sub_locs = runSubBurstNnf(wmeans,1,
        #                                    nblocks,k=1, # patchsize=1 since tiled
        #                                    mask=mask,valMean=valMean,
        #                                    blockLabels=pad_glocs,
        #                                    img_shape = img_shape)

        # print("with grouped, at optimal: ",sub_vals[0,16,16,0].item())
        # print("glocs:\n",pad_glocs[0,0,16,16,:])



        # glocs = glocs[0,0,16,16,:,:]
        # eqs = torch.all(torch.all(sblocks == glocs,dim=-1),dim=-1)
        # print(eqs.shape)
        # print("torch.any(eqs): ",torch.any(eqs))
        # print(torch.where(eqs))

        break

        # exit()

        # -- 5.) re-compute vals from proposed locs --
        ud_names = groups
        print("TYPES")
        print(locs.type())
        print(sub_locs.type())
        print(ud_names.type())
        print(locs.shape)
        print(sub_locs.shape)
        print(ud_names.shape)
        pad_locs = padLocs(locs,nbHalf,'extend')
        sub_locs = sub_locs.type(torch.long)
        print("locs.shape: ",locs.shape)
        print("sub_locs.shape: ",sub_locs.shape)
        print("ud_names.shape: ",ud_names.shape)
        search_locs = update_state_locs(locs,sub_locs,ud_names)
        print("search_locs.shape: ",search_locs.shape)
        print("search_locs[:,0,16,16,0,:]")
        print(search_locs[:,0,16,16,0,:])


        # print("search_blocks.shape: ",search_blocks.shape)
        # print("locs.shape: ",locs.shape)
        # print("sub_locs.shape: ",sub_locs.shape)
        print("search_locs.shape: ",search_locs.shape)
        search_locs = rearrange(search_locs,'t i h w k two -> k i h w t two')
        search_locs = search_locs.contiguous()
        # print("search_locs.shape: ",search_locs.shape)
        prop_vals,prop_locs = runSubBurstNnf(wnoisy,1,nblocks,k=1,
                                           blockLabels=search_locs,
                                           img_shape=img_shape)
        prop_locs = rearrange(prop_locs,'i t h w k two -> t i h w k two')

        # -- 6.) update vals and locs --
        cc_names = center_crop(names[...,0,:,:],img_shape[1:])
        vals,locs,exp_locs = update_state(vals,locs,prop_vals,prop_locs,cc_names,False)
        max_displ = np.max([locs.max().item(),np.abs(locs.min().item())])
        assert max_displ <= nblocks//2, "displacement must remain contained!"


        # -- 7.) rewarp bursts --
        pad_locs = padLocs(locs,nbHalf,'extend')
        nframes,nimages,hP,wP,k,two = pad_locs.shape
        psize = edict({'h':hP,'w':wP})
        # p_exp_locs = padLocs(exp_locs,nbHalf,'extend')
        warped_noisy = warp_burst_from_locs(wnoisy,pad_locs,nblocks,psize)
        warped_clean = warp_burst_from_locs(wclean,pad_locs,nblocks,psize)

    warped_noisy = center_crop(wnoisy,ishape)
    warped_clean = center_crop(wclean,ishape)

    warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    warped_clean = index_along_ftrs(warped_clean,patchsize,c)
    
    if to_flow:
        locs = locs2flow(locs)

    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs,warped_noisy#,warped_clean
