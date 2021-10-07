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


def runBpSearch_rand(noisy, clean, patchsize, nblocks, k = 1,
                     nparticles = 1, niters = 50,
                     valMean = 0., std = None,
                     l2_nblocks = None, l2_valMean=0.,
                     blockLabels=None, ref=None,
                     to_flow=False, fmt=False):


    # -------------------------------
    #
    # ----    error checking     ----
    #
    # -------------------------------

    if l2_nblocks is None: l2_nblocks = nblocks
    assert nparticles == 1, "Only one particle currently supported."
    if std is None: std = torch.std(noisy.reshape(-1)).item()

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    device = noisy.device
    nframes,nimages,c,h,w = noisy.shape
    ishape = [h,w]
    img_shape = [c,h,w]
    psHalf,nbHalf = patchsize//2,nblocks//2
    fPad = 2*(psHalf + nbHalf)
    int_shape = [h-fPad,w-fPad]
    isize = edict({'h':h,'w':w})
    psize = edict({'h':h+2*nbHalf,'w':w+2*nbHalf})
    pshape = [h+psHalf,w+psHalf]
    mask = torch.zeros(h+2*nbHalf,w+2*nbHalf).to(device)
    MAX_SEARCH_FRAMES = 4
    numSearch = min(MAX_SEARCH_FRAMES,nframes-1)
    if np.isclose(valMean,0):
        ps2 = patchsize**2
        t = numSearch + 1
        valMean = (t - 1) / t**2 * std**2 #* (ps2- 2)/ps2


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
    # l2_locs = torch.zeros_like(locs)
    l2_locs = locs

    # # -- 2.) create local search radius from topK locs --
    nframes,nimages,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)
    search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)

    # -- 3.) pad and tile --
    tnoisy = padAndTileBatch(noisy,patchsize,nblocks)
    img_shape[0] = tnoisy.shape[-3]

    # -- 4.) update "val" from "l2" to "burst" @ curr --
    print("l2_locs.shape ",l2_locs.shape)
    l2_flow = locs2flow(l2_locs)
    print("l2_locs @ (16,16): ",l2_locs[:,0,16,16,0,:])
    print("l2_flow @ (16,16): ",l2_flow[:,0,16,16,0,:])
    l2_flow_v2 = l2_flow.clone()
    # l2_flow_v2[0,0,16,16,0,0] = 0
    # l2_flow_v2[2,0,16,16,0,0] = 0
    # l2_flow_v2[0,0,0,0,0,0] = 0
    # l2_flow_v2[2,0,0,0,0,0] = 0
    l2_flow_v2[0,0,:,:,0,0] = l2_flow_v2[0,0,16,16,0,0]
    l2_flow_v2[2,0,:,:,0,0] = l2_flow_v2[2,0,16,16,0,0]
    print("l2_flow_v2 @ (16,16): ",l2_flow_v2[:,0,16,16,0,:])
    print("-"*30)

    print("-"*30)
    print("-"*30)
    l2_locs_stnd = l2_locs
    l2_locs = rearrange(l2_locs,'t i h w 1 two -> i (h w) t two')
    l2_flow = rearrange(l2_flow,'t i h w 1 two -> i (h w) t two')
    l2_flow_v2 = rearrange(l2_flow_v2,'t i h w 1 two -> i (h w) t two')
    vals,e_locs = evalAtLocs(tnoisy, l2_locs_stnd, 1,
                             nblocks, img_shape=img_shape)
    elocs_vals = vals
    print("Mode of Vals: ",mode_vals(vals,int_shape))
    print("Std of Vals: ",torch.std(vals).item())
    print("[evalAtLocs] vals @ (16,16): ",vals[0,16,16,0].item())
    print("-"*30)


    vals,_ = evalAtFlow(tnoisy, l2_flow, 1,
                        nblocks, img_shape = img_shape)
    flow_vals = vals
    print("Mode of Vals: ",mode_vals(vals,int_shape))
    print("Std of Vals: ",torch.std(vals).item())
    print("[flow] vals @ (16,16): ",vals[0,16,16,0].item())
    print("-"*30)


    vals,_ = evalAtFlow(tnoisy, l2_flow_v2, 1,
                        nblocks, img_shape = img_shape)
    flow_v2_vals = vals
    print("Mode of Vals: ",mode_vals(vals,int_shape))
    print("Std of Vals: ",torch.std(vals).item())
    print("[flow_v2] vals @ (16,16): ",vals[0,16,16,0].item())
    print("-"*30)

    vals,_ = evalAtFlow(tnoisy, l2_locs, 1,
                        nblocks, img_shape = img_shape)
    locs_vals = vals
    print("Mode of Vals: ",mode_vals(vals,int_shape))
    print("Std of Vals: ",torch.std(vals).item())
    print("[locs] vals @ (16,16): ",vals[0,16,16,0].item())
    print("-"*30)

    print("-"*30)
    print("-"*30)

    delta = np.abs(flow_vals[0,16,16,0].item() - locs_vals[0,16,16,0].item())
    print("[abs(flow - locs)]: vals @ (16,16) ",delta)
    print("Mode: ",mode_vals(vals,int_shape))
    vals = torch.abs(vals - valMean)
    print("[Post] Mode: ",mode_vals(vals,int_shape))
    print("valMean: ",valMean)
    # assert torch.all(vals > 0), "all 

    search_blocks,_ = getBlockLabels(None,nblocks,torch.long,device,True,t=nframes)
    nsearch = search_blocks.shape[1]
    print("search_blocks.shape ",search_blocks.shape)
    
    # xgrid = [14,15,16,17,14,15,16,17,18,19,9,8,9,8,9,8,9,8]
    # ygrid = [14,14,14,14,15,15,15,15,16,16,17,17,10,11,12,10,11,12,10,11]
    x,y = np.mgrid[3:28,3:28]
    xgrid,ygrid = np.stack(x).ravel(),np.stack(y).ravel()
    obs = []
    for x,y in zip(xgrid,ygrid):
        f = torch.LongTensor([[0,0],[0,0]]).to(device)
        for l in range(nsearch):
            f[0,:] = search_blocks[0,l,:]
            f[1,:] = search_blocks[2,l,:]
            # print("dnoisy [[-1,1],[0,0],[1,-1]]")
            frame1 = tnoisy[0,0,:,x+f[0,0]+nbHalf,y+f[0,1]+nbHalf]
            frame2 = tnoisy[1,0,:,x+nbHalf,y+nbHalf]
            frame3 = tnoisy[2,0,:,x+f[1,0]+nbHalf,y+f[1,1]+nbHalf]
            frames = torch.stack([frame1,frame2,frame3],dim=0)
            ave = torch.mean(frames,dim=0)[None,:]
            cond = f[0,0] == -1 and f[0,1] == 1 and f[1,0] == 1 and f[1,1] == -1
            if cond:
                # print(frames.shape)
                # print(ave.shape)
                # print(f)
                # check = torch.mean((frames[0] - frames[1])**2).item()
                # print(check)
                # check = torch.mean((frames[2] - frames[1])**2).item()
                # print(check)
                delta0 = torch.mean((frames[0] - ave)**2).item()
                # print(delta0)
                delta1 = torch.mean((frames[1] - ave)**2).item()
                # print(delta1)
                delta2 = torch.mean((frames[2] - ave)**2).item()
                # print(delta2)
                # delta = torch.mean((frames - ave)**2).item()
                # print(delta)
                obs.append(delta0)
                obs.append(delta1)
                obs.append(delta2)

                check = torch.mean(((frames[1] - frames[0] +
                                     frames[2] - frames[0])/3.)**2).item()
                # print("c",check)

    obs = np.array(obs).ravel()
    print("obs.shape: ",obs.shape)
    mode = mode_ndarray(obs)
    print("Mean: ",np.mean(obs))
    print("Std: ",np.std(obs))
    print("Mode: ",mode)
    print("-"*80)

    rands = np.random.permutation(len(obs))[:100]
    sub_obs = np.array(obs[rands])
    mode = mode_ndarray(sub_obs)
    print("Mean: ",np.mean(sub_obs))
    print("Std: ",np.std(sub_obs))
    print("Mode: ",mode)
    print("-"*80)

    rands = np.random.permutation(len(obs))[:100]
    sub_obs = np.array(obs[rands])
    mode = mode_ndarray(sub_obs)
    print("Mean: ",np.mean(sub_obs))
    print("Std: ",np.std(sub_obs))
    print("Mode: ",mode)
    print("-"*80)

    rands = np.random.permutation(len(obs))[:100]
    sub_obs = np.array(obs[rands])
    mode = mode_ndarray(sub_obs)
    print("Mean: ",np.mean(sub_obs))
    print("Std: ",np.std(sub_obs))
    print("Mode: ",mode)
    print("-"*80)


    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig,ax = plt.subplots(1,figsize=(8,8))
    sns.kdeplot(obs,label="obs",ax=ax)
    plt.savefig("./bp_search_rand.png",
                transparent=True,dpi=300)
    plt.close("all")

    exit()
        
        # if np.abs(delta-0.50705) < 1e-3:
        #     print(f)
        #     print(delta)
    
    print("dnoisy [[-1,1],[0,0],[1,-1]]")
    frame1 = tnoisy[0,0,:,15+nbHalf,17+nbHalf]
    frame2 = tnoisy[1,0,:,16+nbHalf,16+nbHalf]
    frame3 = tnoisy[2,0,:,17+nbHalf,15+nbHalf]
    frames = torch.stack([frame1,frame2,frame3],dim=0)
    ave = torch.mean(frames,dim=0)[None,:]
    delta = torch.sum((frames - ave)**2).item()/2.
    print(delta)

    print("dnoisy [[1,1],[0,0],[-1,-1]]")
    frame1 = tnoisy[0,0,:,17+nbHalf,17+nbHalf]    
    frame2 = tnoisy[1,0,:,16+nbHalf,16+nbHalf]
    frame3 = tnoisy[2,0,:,15+nbHalf,15+nbHalf]
    frames = torch.stack([frame1,frame2,frame3],dim=0)
    ave = torch.mean(frames,dim=0)[None,:]
    delta = torch.sum((frames - ave)**2).item()/2.
    print(delta)

    print("dnoisy [[-1,0],[0,0],[1,0]]")
    frame1 = tnoisy[0,0,:,15+nbHalf,16+nbHalf]    
    frame2 = tnoisy[1,0,:,16+nbHalf,16+nbHalf]
    frame3 = tnoisy[2,0,:,17+nbHalf,16+nbHalf]
    frames = torch.stack([frame1,frame2,frame3],dim=0)
    ave = torch.mean(frames,dim=0)[None,:]
    delta = torch.sum((frames - ave)**2).item()/2.
    print(delta)

    # dnoisy = torch.sum((tnoisy[0,0,:,15,17] - tnoisy[1,0,:,16,16])**2).item()
    # print(dnoisy)
    # dnoisy = torch.sum((tnoisy[2,0,:,17,15] - tnoisy[1,0,:,16,16])**2).item()
    # print(dnoisy)

    print("dnoisy [[0,1],[0,-1]]")
    print("tnoisy.shape ",tnoisy.shape)
    frame1 = tnoisy[0,0,:,17+nbHalf,15+nbHalf]    
    frame2 = tnoisy[1,0,:,16+nbHalf,16+nbHalf]
    frame3 = tnoisy[2,0,:,15+nbHalf,17+nbHalf]
    frames = torch.stack([frame1,frame2,frame3],dim=0)
    ave = torch.mean(frames,dim=0)[None,:]
    delta = torch.sum((frames - ave)**2).item()
    print(delta)

    # dnoisy = torch.sum((tnoisy[0,0,:,15,16] - tnoisy[1,0,:,16,16])**2).item()
    # print(dnoisy)
    # dnoisy = torch.sum((tnoisy[2,0,:,17,16] - tnoisy[1,0,:,16,16])**2).item()
    # print(dnoisy)


    exit()

    # -- 5.) warp burst to top location --
    pixPad = (tnoisy.shape[-1] - noisy.shape[-1])//2
    plocs = padLocs(locs,pixPad,'extend')
    warped_noisy = warp_burst_from_locs(tnoisy,plocs,1,psize)[0]

    # -- compute search ranges for number of search frames --
    ngroups = numSearch+1
    # search_ranges = create_search_ranges(nblocks,h,w,nsearch,0)
    # search_ranges = torch.LongTensor(search_ranges[:,None]).to(device)
    # search_blocks = compute_search_blocks(search_ranges,0)
    search_blocks,_ = getBlockLabels(None,nblocks,torch.long,device,True,t=ngroups)
    ngroups,nsearch,two = search_blocks.shape
    left = search_blocks[:ngroups//2] 
    right = search_blocks[ngroups//2+1:] 
    search_blocks = torch.cat([search_blocks[[ngroups//2]],left,right],dim=0)
    
    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    print("valMean: ",valMean)
    counts = torch.zeros(nframes)
    for i in range(niters):
        prev_locs = locs.clone()

        # -- 1.) cluster each pixel across time --
        search_frames,names,nuniuqes = temporal_inliers_outliers(tnoisy,warped_noisy,vals,
                                                                 std,numSearch=numSearch)
        # print("Names: ",list(names.cpu().numpy()))
        counts[names] += 1

        # -- 2.) exh srch over a selected frames --
        sub_vals,sub_locs = runBurstNnf(search_frames, 1, nblocks, k=1,
                                        blockLabels=search_blocks,
                                        img_shape=img_shape,valMean=valMean)
        # print("Num Uniques: ",nuniuqes[:,16,16])
        sub_vals = sub_vals / center_crop(nuniuqes,ishape)[...,None]
        sub_vals = torch.abs(sub_vals - valMean)

        # -- 3.) update vals and locs --
        vals,locs = update_state_outliers(vals,locs,sub_vals,
                                          sub_locs,names,False)
        max_displ = torch.abs(locs).max().item()
        assert max_displ <= nbHalf, "displacement must remain contained!"
        print("vals @ (16,16): ",vals[0,16,16,0])
        print("sub_vals @ (16,16): ",sub_vals[0,16,16,0])

        # -- 4.) rewarp bursts --
        pad_locs = padLocs(locs,nbHalf,'extend')
        nframes,nimages,hP,wP,k,two = pad_locs.shape
        warped_noisy_old = warped_noisy.clone()
        warped_noisy = warp_burst_from_locs(tnoisy,pad_locs,nblocks,psize)[0]

        delta = torch.sum(torch.abs(prev_locs - locs)).item()
        # print("Delta: ",delta)

        # delta = torch.sum(torch.abs(prev_locs[:,0,16,16,0] - locs[:,0,16,16,0])).item()
        # delta = torch.sum(torch.abs(warped_noisy_old[...,16,16] - \
        #                             tnoisy[...,16,16])).item()
        # delta = torch.sum(torch.abs(warped_noisy_old[...,16+1,16+1] - \
        #                             warped_noisy[...,16+1,16+1])).item()
        # print("Delta: ",delta)

    # -------------------------------
    #
    # --    finalizing outputs     --
    #
    # -------------------------------
    # print("Counts: ",counts)

    # exit()
    # -- get the output image from tiled image --
    warped_noisy = center_crop(warped_noisy,ishape)
    warped_noisy = index_along_ftrs(warped_noisy,patchsize,c)
    
    # -- convert "locs" to "flow" --
    if to_flow:
        locs = locs2flow(locs)

    # -- reformat for experiment api --
    if fmt:
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two')

    return vals,locs,warped_noisy#,warped_clean
