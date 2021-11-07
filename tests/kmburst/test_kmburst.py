
# -- python imports --
import time
import torch
import faiss
import contextlib
import numpy as np
from PIL import Image
from einops import rearrange,repeat
from easydict import EasyDict as edict
import scipy.stats as stats

# -- project --
from pyutils import save_image,get_img_coords

# -- plotting imports --
import matplotlib
matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from pyutils import save_image
from pyutils.images import images_to_psnrs,images_to_psnrs_crop
from align import nnf 
from align.xforms import pix_to_blocks,align_from_flow,pix_to_flow
from datasets.transforms import get_dynamic_transform

# -- faiss-python imports --
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from torch_utils import swig_ptr_from_FloatTensor,using_stream
import nnf_utils as nnf_utils
import bnnf_utils as bnnf_utils
import sub_burst as sbnnf_utils
from bp_search import runBpSearch
from nnf_share import padAndTileBatch,padBurst,tileBurst
from warp_utils import warp_burst_from_locs,warp_burst_from_pix,warp_burst_from_flow
from warp_utils import pix2locs,locs2flow,flow2locs
from kmb_search import runKmSearch,compute_mode_pairs
from kmb_search.testing.utils import compute_gt_burst,set_seed

def exp_setup():
    # seed = 234
    # seed = 345
    seed = 456
    # seed = 678
    # seed = 789
    set_seed(seed)
    
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    h,w,c = 16,16,3
    # h,w,c = 17,17,3
    # h,w,c = 512,512,3
    # h,w,c = 256,256,3
    # h,w,c = 128,128,3
    # h,w,c = 64,64,3
    # h,w,c = 16,16,3
    # h,w,c = 15,15,2
    # h,w,c = 47,47,3
    # h,w,c = 32,32,3
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    # ps,nblocks = 11,10
    patchsize = 3
    nblocks = 3
    k = 3#(nblocks**2)**2
    gpuid = 0
    return seed,h,w,c,patchsize,nblocks,k,gpuid

def test_burst_nnf_sample():

    # ---------------------------
    #
    #        EXP SETTINGS 
    #
    # ---------------------------
    seed,h,w,c,patchsize,nblocks,k,gpuid = exp_setup()

    # -- derived variables --
    ps = patchsize
    pad = int(ps//2)+int(nblocks//2)
    pads = (pad,pad,pad,pad)
    
    # -- apply dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = 15
    dynamic_info.ppf = 1
    dynamic_info.frame_size = [h,w]
    dyn_xform = get_dynamic_transform(dynamic_info,None)
    t = dynamic_info.nframes

    # -- outputs --
    vals = edict()
    locs = edict()
    runtimes = edict()
    
    # ----------------------------
    # 
    #    Create Images & Flow
    # 
    # ----------------------------

    # -- sample data --
    image = np.random.rand(3*h,3*w,c).astype(np.float32)
    # image[1::2,::2] = 1.
    # image[::2,1::2] = 1.
    image = np.uint8(image*255.)
    imgPIL = Image.fromarray(image)
    dyn_result = dyn_xform(imgPIL)
    burst = dyn_result[0][:,None].to(gpuid)
    flow =  dyn_result[-2]
    device = burst.device
    ref = t//2

    # -- format data --
    std = 50.
    noise = np.random.normal(loc=0,scale=std/255.,size=(t,1,c,h,w)).astype(np.float32)
    clean = burst.clone()
    repimage = repeat(clean[ref],'1 c h w -> t 1 c h w',t=t).clone()
    burst += torch.FloatTensor(noise).to(device)
    block = np.c_[-flow[:,1],flow[:,0]] # (dx,dy) -> (dy,dx) with "y" [0,M] -> [M,0]
    block_gt = block
    flow_gt = flow
    print("-- block --")
    print(block)
    print("-- flow --")
    print(flow)
    nframes,nimages,c,h,w = burst.shape
    isize = edict({'h':h,'w':w})
    flows_hw = repeat(flow,'t two -> 1 h w t two',h=h,w=w)
    flows = repeat(flow,'t two -> 1 p t two',p=h*w)
    blocks_gt = repeat(block,'t two ->  two t h w',h=h,w=w)
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    indices_gt = torch.LongTensor(blocks_gt).to(device) + coords
    aligned = align_from_flow(clean,flows,0,isize=isize)
    save_image("tkmb_burst.png",burst)
    save_image("tkmb_clean.png",clean)
    save_image("tkmb_repimage.png",repimage)
    save_image("tkmb_aligned.png",aligned)
    psnrs = images_to_psnrs_crop(aligned,repimage,ps)
    print("GT PSNRS [aligned vs repimage]: ",psnrs)
    psnrs = images_to_psnrs(clean,repimage)
    print("GT PSNRS [clean v.s. repimage: ",psnrs)
    assert dynamic_info.ppf <= (nblocks//2), "nblocks is too small for ppf"

    # -- add padding --
    t,b,c,h,w = burst.shape
    burstPad,offset = bnnf_utils.padBurst(burst[:,0],(c,h,w),ps,nblocks)[:,None],2
    print("original burst ",burst.shape)
    print("pad no tile ",burstPad.shape)

    # -- get block labels --
    # blockLabels,_ = bnnf_utils.getBlockLabels(None,nblocks,np.float32,
    #                                           'cpu',False,nframes)

    # ------------------------------
    #
    #    Extra Compute to "Burn-in"
    #
    # ------------------------------

    valMean = 1.
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(burst,
                                              patchsize, nblocks,
                                              1, valMean = valMean,
                                              blockLabels=None)
    
    # ------------------------------
    #
    #    Compute L2-Local Search
    #
    # ------------------------------

    valMean = 0.#compute_mode_pairs(std/255.,c,patchsize)
    print("valMean: ",valMean)
    start_time = time.perf_counter()
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(burst, patchsize,
                                              nblocks, 1,
                                              valMean = valMean,
                                              blockLabels=None)
    runtimes.L2Local = time.perf_counter() - start_time
    vals.L2Local = nnf_vals[:,0]
    locs.L2Local = pix2locs(nnf_locs)
    # print("locs.L2Local.shape: ",locs.L2Local.shape)
    warp = warp_burst_from_locs(clean,locs.L2Local)[0]    
    psnrs = images_to_psnrs_crop(warp[:,0],repimage[:,0],ps)
    print("-- [L2Local.psnrs] --")
    print(psnrs)
    save_image("tkmb_l2_warp.png",warp)

    # ------------------------------------------
    #
    #     Compute FAISS KmBurstNnf
    #
    # ------------------------------------------

    print("-"*30)
    print("KMeans Burst")
    print("-"*30)
    gt_info = {'indices':indices_gt,'clean':clean[:,0].transpose(0,1)}
    gt_dist = 0.
    start_time = time.perf_counter()    
    _vals,_locs = runKmSearch(burst, patchsize,nblocks, k = k,
                              std = std/255.,search_space=None,
                              ref=None,mode="python",gt_info=gt_info)
    runtimes.KmBurst = time.perf_counter() - start_time
    print("_vals.shape: ",_vals.shape)
    print("_locs.shape: ",_locs.shape)
    # print(_vals[0,0,4:6,4:6])
    # print(_locs[0,0,:,4,4,:])
    # tmp = _locs[...,0].clone()
    # _locs[...,0] = _locs[...,1]
    # _locs[...,1] = tmp
    # vals.shape: (i,k,h,w)
    # locs.shape: (i,k,t,h,w,2)
    vals.KmBurst = _vals[0,0]
    locs.KmBurst = _locs[0,0]
    # print(_locs[0,0,:,4,4,:])
    wlocs = rearrange(_locs,'i k t h w two -> t i h w k two')
    warp = warp_burst_from_flow(clean,wlocs)[0]
    # warp = warp_burst_from_locs(clean,wlocs)[0]    
    psnrs = images_to_psnrs_crop(warp[:,0],repimage[:,0],ps)
    print("-- [KmBurst.psnrs] --")
    print(psnrs)
    save_image("tkmb_warp.png",warp)

    # -- delta flow --
    print("-- compare with gt flow --")
    flows_kmb = locs.KmBurst
    flows_hw = rearrange(flows_hw,'1 h w t two -> t h w two')
    print(flows_kmb.shape,flows_hw.shape)
    for t in range(flows_hw.shape[0]):
        delta = torch.abs(flows_hw[t] - flows_kmb[t])
        delta = delta.type(torch.float).mean()
        print("[%d] Delta: %2.3f" %(t,delta))

    # -- for exps --
    print("locs.KmBurst.shape: ",locs.KmBurst.shape)
    t,h,w,two = locs.KmBurst.shape
    flows_fmt = rearrange(locs.KmBurst,'t h w two -> 1 (h w) t two').clone()
    isize = edict({'h':h,'w':w})
    # flows = pix_to_flow(locs_fmt)
    
    aligned = align_from_flow(clean,flows_fmt,0,isize=isize)
    # aligned = align_from_flow(burst,flows_fmt,patchsize,isize=isize)
    psnrs = images_to_psnrs_crop(aligned[:,0],repimage[:,0],ps)
    print("-- [exp-pipe] [KmBurst.psnrs] --")
    print(psnrs)

    print(" -- locs.KmBurst --")
    print(locs.KmBurst[:,8,9,:])

    return

    # locs.KmBurst = _locs[0]
    print(locs.KmBurst.shape)
    print(_locs.shape)
    print("[locs.KmBurst] @ (0,0): \n",locs.KmBurst[:,0,0,0])
    print("[locs.KmBurst] @ (0,1): \n",locs.KmBurst[:,0,1,0])
    print("[locs.KmBurst] @ (16,16): \n",locs.KmBurst[:,16,16,0])

    print("vals.shape ",vals.KmBurst.shape)
    print("[vals.KmBurst] @ (0,0): \n",vals.KmBurst[0,0].item())
    print("[vals.KmBurst] @ (0,1): \n",vals.KmBurst[0,1].item())
    print("[vals.KmBurst] @ (16,16): \n",vals.KmBurst[16,16].item())

    print("-"*30)

    # ------------------------------
    #
    #    Compute BP Search
    #
    # ------------------------------

    valMean = 0.
    img_shape = list(burst.shape[-3:])
    img_shape = list(img_shape)
    print("burst.shape ",burst.shape)
    burstPad_0 = padBurst(burst[:,0],img_shape,patchsize,nblocks)
    print("padded burst.shape ",burstPad_0.shape)
    wburst = tileBurst(burstPad_0,h,w,patchsize,nblocks)[:,None]
    # wburst = padAndTileBatch(burst,patchsize,nblocks)
    img_shape[0] = wburst.shape[-3]
    print("wburst.shape ",wburst.shape)
    start_time = time.perf_counter()
    # _vals,_locs,_ = runBpSearch(burst, burst, patchsize, nblocks,
    #                             k=k, valMean = valMean,
    #                             blockLabels=None,
    #                             search_type="cluster_approx")
    _vals,_locs = sbnnf_utils.runBurstNnf(wburst, 1, nblocks,
                                          k=3, valMean = valMean,
                                          blockLabels=None,
                                          img_shape=img_shape)
    # _vals,_locs = sbnnf_utils.runBurstNnf(burst, patchsize, nblocks,
    #                                       k=3, valMean = valMean,
    #                                       blockLabels=None,
    #                                       img_shape=None)
    runtimes.BpSearch = time.perf_counter() - start_time
    vals.BpSearch = _vals[0]
    locs.BpSearch = _locs[:,0]
    # locs.BpSearch = _locs[0]

    # -----------------------------------
    #
    #    Compute BurstNNF @ Given Flow
    #
    # -----------------------------------

    flow_fmt = rearrange(flow,'t two -> 1 1 t two')
    mode = bnnf_utils.evalAtFlow(burst, flow_fmt, patchsize,
                                 nblocks, return_mode=True,
                                 tile_burst=False)
    h,w = burst.shape[-2:]
    locs_img = repeat(flow,'t two -> t 1 h w 1 two',h=h,w=w)
    print("locs_img.shape ",locs_img.shape)
    # sub_vals,e_locs = sbnnf_utils.evalAtLocs(burst,
    #                                          locs_img, patchsize,
    #                                          nblocks,
    #                                          img_shape=img_shape)
    sub_vals,e_locs = sbnnf_utils.evalAtLocs(wburst,
                                             locs_img, 1,
                                             nblocks,
                                             img_shape=img_shape)
                                             

    print("mode.shape ",mode.shape)
    for h_i in range(h):
        for w_i in range(w):
            break
            pix_hw = [h_i,w_i]
            sub_hw = sub_vals[0,h_i,w_i,0]
            mode_hw = mode[h_i,w_i].item()
            # print("Optimal Values: [median]: %2.5e" % (mode_hw))
            # pix_hw = [1,1]
            # pix_hw = [h//2,w//2]
            # pix_hw = [h-1,w-1]
            gt_dist = compute_gt_burst(burstPad,pix_hw,block,ps,nblocks)
            # print("GT :",gt_dist)
            # print(h_i,w_i)
            diff = np.abs(gt_dist - mode_hw).item()
            # if diff > 1e-5:
            #     print(h_i,w_i,mode_hw,gt_dist,diff)
            #     assert mode_hw > gt_dist," testing ineq"
            assert np.abs(gt_dist - mode_hw) < 1e-5, "must be equal for all pix."
            assert np.abs(gt_dist - sub_hw) < 1e-5, "must be equal for all pix."
            cond1 = h_i == h//2 and w_i == w//2
            cond2 = h_i == 0 and w_i == 0
            conds = cond1 or cond2
            if conds:
                print("Optimal Values: [median]: %2.5e" % (mode_hw))
                print("GT: ",gt_dist)
    gt_dist = 0.


    # -----------------------------------
    #
    #      Compute FAISS BurstNnf     
    #
    # -----------------------------------

    valMean = gt_dist#3.345
    start_time = time.perf_counter()
    _vals,_locs = bnnf_utils.runBurstNnf(burst, patchsize, nblocks, k = k,
                                         valMean = valMean, blockLabels=None,
                                         ref=None,
                                         to_flow=False,
                                         tile_burst=False)
    runtimes.Burst = time.perf_counter() - start_time
    vals.Burst = _vals[0]
    locs.Burst = _locs[:,0]

    # -----------------------------------
    #
    #        Compare Outputs # 1
    #
    # -----------------------------------
    
    print("cuda vals!")
    print("Our-Local runtime [No \"Unfold\" + Local Search]: ",runtimes.L2Local)
    print("Our-Burst runtime [No \"Unfold\" + Local Search]: ",runtimes.Burst)
    print("Our-KmBurst runtime [No \"Unfold\" + Local Search]: ",runtimes.KmBurst)
    print("Our-BpSearch runtime [No \"Unfold\" + Local Search]: ",runtimes.BpSearch)

    # -----------------------------------
    #
    #      Compare _Centered Pixels_
    #
    # -----------------------------------

    # pix_hw = [1,1]
    # pix_hw = [h-1,w-1]
    pix_hw = [h//2,w//2]
    # pix_hw = [h//2+3,w//2-4]
    pix_hw_list = [[h//2,w//2],[h//2-1,w//2],[h//2,w//2-1],[h//2+1,w//2],
                   [h//2,w//2+1],[h//2-1,w//2+1],[h//2+1,w//2-1],[0,0],[0,1],
                   [1,0],[1,1],[h-1,w-1],[h-2,w-1],[h-1,w-2],[h-2,w-2],[h-3,w-3]]
    interior_pad = nblocks//2+patchsize//2+3
    neqImage = torch.zeros(h,w)
    for i in range(interior_pad,h-interior_pad):
        for j in range(interior_pad,w-interior_pad):
            for key1 in vals:
                if key1 == "L2Local": continue
                v_key1 = vals[key1][i,j]
                l_key1 = locs[key1][:,i,j,0]
                for key2 in vals:
                    if key2 == "L2Local": continue
                    if key1 == key2: continue
                    v_key2 = vals[key2][:,i,j,0]
                    l_key2 = locs[key2][:,i,j,0]

                    delta = torch.sum(torch.abs(v_key1 - v_key2)).item()
                    if delta > 1e-5:
                        if key2 == "BpSearch":
                            print(i,j,key1,key2,v_key1,v_key2,l_key1,l_key2)
                            neqImage[i,j] = 1.
                        elif key1 != "BpSearch" and key2 != "BpSearch":
                            print(i,j,v_key1,v_key2,delta,key1,key2)
                    assert delta < 1e-5, "The outputs must agree on interior!"
    save_image(neqImage[None,:],f"neqImage_{seed}.png")


    for pix_hw in pix_hw_list:
        print(pix_hw)
        gt_dist = compute_gt_burst(burstPad,pix_hw,block,ps,nblocks)
        print("Our-L2-Local Output: ",vals.L2Local[pix_hw[0],pix_hw[1]].item())
        print("Burst-L2-Local Output: ",vals.Burst[pix_hw[0],pix_hw[1]].item())
        print("KmBurst-L2-Local Output: ",vals.KmBurst[pix_hw[0],pix_hw[1]].item())
        print("BpSearch Output: ",vals.BpSearch[pix_hw[0],pix_hw[1]].item())
        print("GT Output: ",gt_dist)

    
        print("[Locs] Our-L2-Local: ",locs.L2Local[:,pix_hw[0],pix_hw[1],0])
        print("[Locs] Burst-L2-Local: ",locs.Burst[:,pix_hw[0],pix_hw[1],0])
        print("[Locs] KmBurst-L2-Local: ",locs.KmBurst[:,pix_hw[0],pix_hw[1],0])
        print("[Locs] BpSearch: ",locs.BpSearch[:,pix_hw[0],pix_hw[1],0])
        # print("GT Output: ",gt_dist)


    pix_hw = [0,0]
    bp_top1_loc = locs.BpSearch[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    b_top1_loc = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    l2_top1_loc = locs.L2Local[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("BpSearch Loc: ",bp_top1_loc)
    print("Burst Loc: ",b_top1_loc)
    print("L2 Loc: ",l2_top1_loc)

    pix_hw = [16,16]
    bp_top1_loc = locs.BpSearch[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    b_top1_loc = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    l2_top1_loc = locs.L2Local[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("BpSearch Loc: ",bp_top1_loc)
    print("Burst Loc: ",b_top1_loc)
    print("L2 Loc: ",l2_top1_loc)

    pix_hw = [h//2,w//2]
    # pix_hw = [h//2,w//2]
    vBurst = vals.Burst[pix_hw[0],pix_hw[1]].item()
    vKmBurst = vals.KmBurst[pix_hw[0],pix_hw[1]].item()
    iBurstValid = np.where(vBurst < 1e10)
    iKmBurstValid = np.where(vKmBurst < 1e10)
    assert np.all(iBurstValid[0] == iKmBurstValid[0]),"Equal OOB"
    # assert np.all(iBurstValid[1] == iKmBurstValid[1]),"Equal OOB"


    vBurst = vBurst[ iBurstValid ]
    vKmBurst = vKmBurst[ iKmBurstValid ]
    neqLocs = np.where(vBurst - vKmBurst)
    blockIndicesNeq = iBurstValid[0][neqLocs[0]]
    assert len(blockIndicesNeq) == 0, "Both burst types must be equal."
    # print("blockLabels")
    # print(blockLabels.shape)
    # print(blockLabels[:,blockIndicesNeq])
    # print(blockLabels[:,-5:])

    # print(vals.Burst[pix_hw[0],pix_hw[1]])
    # print(locs.Burst[pix_hw[0],pix_hw[1]])
    bp_top1_loc = locs.BpSearch[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    loc_top1 = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("-"*20)
    # print(vals.Burst[pix_hw[0]-1,pix_hw[1]-1])
    # print(vals.Burst[pix_hw[0]-1,pix_hw[1]+1])
    # print(vals.Burst[pix_hw[0]+1,pix_hw[1]-1])
    # print(vals.Burst[pix_hw[0]+1,pix_hw[1]+1])
    print("-"*20)

    output = []
    output.append(vals.Burst[pix_hw[0],pix_hw[1]].cpu().numpy())
    # output.append(vals.Burst[pix_hw[0]-1,pix_hw[1]-1].cpu().numpy())
    # output.append(vals.Burst[pix_hw[0]-1,pix_hw[1]+1].cpu().numpy())
    # output.append(vals.Burst[pix_hw[0]+1,pix_hw[1]-1].cpu().numpy())
    # output.append(vals.Burst[pix_hw[0]+1,pix_hw[1]+1].cpu().numpy())
    output = np.array(output)[:,0]

    print(flow.shape,block.shape)
    gt_dist = compute_gt_burst(burstPad,pix_hw,block,ps,nblocks)
    top1_val = vals.Burst[pix_hw[0],pix_hw[1]].item()
    top1_loc = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("Burst Output @ Top1: ",top1_val)
    print("GT-Burst Output @ GT Flow: ",gt_dist)
    print("Burst Output @ Top1: ",top1_loc)
    print("GT Block",block)
    print("BpSearch Output @ Top1: ",bp_top1_loc)
    
    assert np.sum(np.abs(gt_dist - top1_val)/(gt_dist+1e-8)) < 1e-6, "vals must equal."
    assert np.sum(np.abs(block - top1_loc)) < 1e-6, "locs must equal."

    flow_bll = np.array([[1,1],[1,1],[1,1]])
    # flow_bll = locs[:,pix_hw[0],pix_hw[1],0]
    # print("GT-Burst Output @ BLL Flow [k = 0]: ",
    #       compute_gt_burst(burst,pix_hw,flow_bll,ps))
    # flow_bll = np.array([[1,0],[1,0],[1,0]])
    # print("GT-Burst Output @ BLL Flow [k = 1]: ",
    #       compute_gt_burst(burst,pix_hw,flow_bll,ps))
    # flow_bll = np.array([[1,-1],[1,-1],[1,-1]])
    # print("GT-Burst Output @ BLL Flow [k = 2]: ",
    #       compute_gt_burst(burst,pix_hw,flow_bll,ps))
    test = []
    blockLabels,_ = bnnf_utils.getBlockLabels(None,nblocks,np.float32,'cpu',False,nframes)
    for bk in range(blockLabels.shape[1]):
        flow_bll = blockLabels[:,bk]
        dist = compute_gt_burst(burstPad,pix_hw,flow_bll,
                                ps,nblocks)
        test.append(dist)
        # print(f"GT-Burst Output @ BLL Flow [k = {bk}]: ",dist)

    test = np.stack(test)
    # print("-="*30)
    # print(test)
    # print(output.shape)
    oH,oW = output.shape
    locs = []
    dists = np.zeros((oH,oW,oW))
    for j1 in range(output.shape[1]):
        locs_j1 = []
        for i in range(output.shape[0]):
            for j0 in range(output.shape[1]):
                o = output[i,j0]
                t = test[j1]
                d = np.abs(o - t)
                dists[i,j0,j1] = d
                if d < 1e-3:
                    locs_j1.append([i,j0])
        locs_j1 = np.array(locs_j1)
        locs.append(locs_j1)
    # print(locs)
    # print(dists)
    print("SUCCESS!")
    return

    # -----------------------------------
    #
    #        Compare _Boundaries_
    #
    # -----------------------------------

    check_xy = [0,0]
    print("L2-Local Output: ",nnf_vals[:,check_xy[0],check_xy[1],:])
    print("Burst-L2-Local Output: ",vals[:,check_xy[0],check_xy[1],:])
    check_xy = [1,1]
    print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
    check_xy = [h//2,w//2]
    print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
    check_xy = [-2,-2]
    print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
    check_xy = [-1,-1]
    print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
    print("Zero Check: ",np.any(vals==0))
    print("Nan Check: ",np.any(np.isnan(vals)))
    
    # offset = ps//2
    print(nnf_locs[0,0,h//2,w//2,0])
    # nnf_locs_x = nnf_locs[0,0,offset:-offset,offset:-offset,0,1] \
    #     - np.arange(w)[None,:] - offset
    # nnf_locs_y = nnf_locs[0,0,offset:-offset,offset:-offset,0,0] \
    #     - np.arange(h)[:,None] - offset
    nnf_locs_x = nnf_locs[0,0,:,:,0,1] \
        - np.arange(w)[None,:]
    nnf_locs_y = nnf_locs[0,0,:,:,0,0] \
        - np.arange(h)[:,None]
    nnf_locs = np.stack([nnf_locs_y,nnf_locs_x],axis=-1)
    intIdx = slice(nblocks//2+ps//2,-(nblocks//2+ps//2))
    print("NNF vs Ours [locs]: ", np.sum(np.abs(locs[intIdx,intIdx,0] -\
                                                nnf_locs[intIdx,intIdx])))
    print(locs[h//2:h//2+2,w//2:w//2+2,0])
    print(nnf_locs[h//2:h//2+2,w//2:w//2+2])
    
    # -----------------------------------
    #
    #        Compare Outputs # 2
    #
    # -----------------------------------
    
    # blockLabelsInt = (blockLabels - 1).astype(np.int32)
    # print(np.sum(np.abs(bfnnf_ref_image[:,1,1] - ref_image[:,0,0])))
    # print(np.sum(np.abs(target_image[:,1,1] - nnf_target[:,0,0].numpy())))
    blockLabelsInt = (blockLabels).astype(np.int32)
    
    # print(blockLabelsInt)
    tol = 1e-4
    print(ps)
    offset = pad-ps//2 # offset for computing GT since we padded images
    
    # vals[vals>10**4]=np.inf
    # print(vals[0,:,0])
    
    for _xstart in np.arange(0,w):
        for _ystart in np.arange(0,h):
            # print(xstart,ystart)
            xstart = _xstart + offset
            ystart = _ystart + offset
    
            res = []
            i = 0
            ref_xy = bfnnf_ref_image[:,xstart:xstart+ps,
                                     ystart:ystart+ps]
            for i in range(blockLabelsInt.shape[0]):
                x,y = blockLabelsInt[i,:]
    
                x_start = xstart + x
                y_start = ystart + y
                x_end = xstart + x + ps
                y_end = ystart + y + ps
                # print(i,(x_start,x_end),(y_start,y_end))
                if x_start < 0 or y_start < 0:
                    # print("continue")
                    res.append(np.inf)
                    continue
                if x_end > target_image.shape[1]:
                    # print("continue")
                    res.append(np.inf)
                    continue
                if y_end > target_image.shape[2]:
                    # print("continue")
                    res.append(np.inf)
                    continue
                tgt_xy = target_image[:,x_start:x_end,
                                      y_start:y_end]
                ref_xy = torch.Tensor(ref_xy)
                tgt_xy = torch.Tensor(tgt_xy)
                # ref_xy = nnf_ref[:,xstart+pad:xstart+ps+pad,
                #                  ystart+pad:ystart+ps+pad]
                # tgt_xy = nnf_target[:,xstart+x+pad:xstart+x+ps+pad,
                #                     ystart+y+pad:ystart+y+ps+pad]
        
                # ref_xy = ref_xy[0,0,0]
                # tgt_xy = tgt_xy[0,0,0]
                res.append(float(torch.sum(torch.pow( ref_xy - tgt_xy , 2)).item()))
            # res = np.sort(np.array(res))
            val = vals[_xstart,_ystart,:]
            val = np.sort(val)
            loc = locs[_xstart,_ystart,:]
            val[val>10**3] = np.inf
            order = blockLabels[np.argsort(res),:][:len(val)]
            res = np.sort(res)
            res = np.nan_to_num(res,posinf=0.)[:len(val)]
            # print(order[:len(val)],loc)
            # print(res - val)
            # print("-"*10)
            def assert_msg():
                msg = "res" + str(res) + "\n\n"
                msg += "GT order " + str(order) + "\n\n"
                msg += "val" + str(val) + "\n\n"
                msg += "loc" + str(loc) + "\n\n"
                msg += "index: " + str(i)+ "\n\n"
                msg += "(x,y): ({},{})".format(xstart,ystart)+ "\n\n"
                msg += "(x,y): ({},{})".format(_xstart,_ystart)+ "\n\n"
                return msg
            msg = assert_msg()
            if xstart == (h//2) and ystart == (w//2):
                print(msg)
            assert np.mean(np.abs(val-res)) < tol, ("Must be equal. " + msg)
    print("SUCCESS! :D")
    
        
def test_burst_nnf():
    for i in range(1):
        test_burst_nnf_sample()
    
if __name__ == "__main__":
    test_burst_nnf()
