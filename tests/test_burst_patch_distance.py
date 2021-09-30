
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

# -- plotting imports --
import matplotlib
matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from pyutils import save_image
from align import nnf 
from align.xforms import pix_to_blocks,align_from_flow
from datasets.transforms import get_dynamic_transform

# -- faiss-python imports --
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from torch_utils import swig_ptr_from_FloatTensor,using_stream
import nnf_utils as nnf_utils
import bnnf_utils as bnnf_utils
import sub_burst as sbnnf_utils


def compute_gt_burst(burst,pix_hw,flow,ps,nblocks):
    psHalf = ps//2
    padOffset = nblocks//2
    patches,patches_ave = [],[]
    nframes,nimages,c,h,w = burst.shape
    for t in range(nframes):
        flow_t = flow[t]

        # startH = pix_hw[1]-flow_t[1]# - psHalf +0# dy
        startH = pix_hw[0] + flow_t[0] + padOffset # dy
        endH = startH + ps
        sliceH = slice(startH,endH)

        # startW = pix_hw[0]+flow_t[0]# - psHalf +0# dx
        startW = pix_hw[1] + flow_t[1] + padOffset # dx
        endW = startW + ps
        sliceW = slice(startW,endW)

        # print(pix_hw,startH,startW,burst.shape)
        patch_t = burst[t,0,:,sliceH,sliceW]
        patches.append(patch_t)


    patches = torch.stack(patches)
    ave = torch.sum(patches,dim=0)/nframes
    # ave = torch.zeros_like(ave)
    diff = torch.sum((patches - ave)**2/nframes).item()
    return diff

def test_burst_nnf_sample():

    # -- settings --
    
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    # h,w,c = 16,16,3
    # h,w,c = 17,17,3
    h,w,c = 512,512,3
    # h,w,c = 256,256,3
    # h,w,c = 128,128,3
    # h,w,c = 64,64,3
    # h,w,c = 15,15,2
    # h,w,c = 47,47,3
    # h,w,c = 32,32,3
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    # ps,nblocks = 11,10
    patchsize = 11
    nblocks = 3
    k = 3#(nblocks**2)**2
    gpuid = 0

    # -- derived variables --
    ps = patchsize
    pad = int(ps//2)+int(nblocks//2)
    pads = (pad,pad,pad,pad)
    
    # -- apply dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = 3
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
    image = np.random.rand(h,w,c).astype(np.float32)
    image[1::2,::2] = 1.
    image[::2,1::2] = 1.
    image = np.uint8(image*255.)
    imgPIL = Image.fromarray(image)
    dyn_result = dyn_xform(imgPIL)
    burst = dyn_result[0][:,None].to(gpuid)
    flow =  dyn_result[-2]
    device = burst.device

    # -- format data --
    print(burst.shape)
    print(burst.max(),burst.min())
    noise = np.random.normal(loc=0,scale=0.2,size=(t,1,c,h,w)).astype(np.float32)
    # burst += torch.FloatTensor(noise).to(device)
    block = np.c_[-flow[:,1],flow[:,0]] # (dx,dy) -> (dy,dx) with "y" [0,M] -> [M,0]
    save_image("burst.png",burst)
    nframes,nimages,c,h,w = burst.shape
    isize = edict({'h':h,'w':w})
    flows = repeat(flow,'t two -> 1 p t two',p=h*w)
    aligned = align_from_flow(burst,flows,patchsize,isize=isize)
    assert dynamic_info.ppf <= (nblocks//2), "nblocks is too small for ppf"

    # -- add padding --
    t,b,c,h,w = burst.shape
    burstPad,offset = bnnf_utils.padBurst(burst[:,0],(c,h,w),ps,nblocks)[:,None],2
    print(burst.shape)
    print(burstPad.shape)

    # -- get block labels --
    blockLabels,_ = bnnf_utils.getBlockLabels(None,nblocks,np.float32,
                                              'cpu',False,nframes)

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

    valMean = 0.
    start_time = time.perf_counter()
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(burst,
                                              patchsize, nblocks,
                                              1, valMean = valMean,
                                              blockLabels=None)
    runtimes.L2Local = time.perf_counter() - start_time
    vals.L2Local = nnf_vals[:,0]
    locs.L2Local = nnf_locs[:,0]

    # ------------------------------
    #
    #    Compute BP Search
    #
    # ------------------------------

    # valMean = 0.
    # start_time = time.perf_counter()
    # nnf_vals,nnf_locs = sbnnf_utils.runBpSearch(burst, patchsize, nblocks,
    #                                             1, valMean = valMean,
    #                                             blockLabels=None)
    # runtimes.BpSearch = time.perf_counter() - start_time
    # vals.BpSearch = nnf_vals[:,0]
    # locs.BpSearch = nnf_locs[:,0]

    # -----------------------------------
    #
    #    Compute BurstNNF @ Given Flow
    #
    # -----------------------------------

    flow_fmt = rearrange(flow,'t two -> 1 1 t two')
    mode = bnnf_utils.evalAtFlow(burst, flow_fmt, patchsize,
                                 nblocks, return_mode=True,
                                 tile_burst=False)
    print("mode.shape ",mode.shape)
    for h_i in range(h):
        for w_i in range(w):
            pix_hw = [h_i,w_i]
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
            if h_i == h//2 and w_i == w//2:
                print("Optimal Values: [median]: %2.5e" % (mode_hw))
                print("GT: ",gt_dist)
            break

    # ------------------------------------------
    #
    #     Compute FAISS SubBurstNnf
    #
    # ------------------------------------------

    valMean = gt_dist
    start_time = time.perf_counter()
    _vals,_locs = bnnf_utils.runBurstNnf(burst, patchsize, nblocks, k = k,
                                         valMean = valMean, blockLabels=None, ref=None)
    runtimes.SubBurst = time.perf_counter() - start_time
    vals.SubBurst = _vals[0][None,:] # include nframes 
    locs.SubBurst = _locs[0]
    print("[shapes of BurstNnf]: ")
    print("vals.shape: ",vals.SubBurst.shape)
    print("locs.shape: ",locs.SubBurst.shape)
    print(locs.SubBurst[:,8,8,0]) 

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
                                         tile_burst=True)
    runtimes.Burst = time.perf_counter() - start_time
    vals.Burst = _vals[0][None,:] # include nframes 
    locs.Burst = _locs[0]
    print("[shapes of BurstNnf]: ")
    print("vals.shape: ",vals.Burst.shape)
    print("locs.shape: ",locs.Burst.shape)
    print(locs.Burst[:,8,8,0]) 

    # -----------------------------------
    #
    #        Compare Outputs # 1
    #
    # -----------------------------------
    
    print("cuda vals!")
    print("Our-Local runtime [No \"Unfold\" + Local Search]: ",runtimes.L2Local)
    print("Our-Burst runtime [No \"Unfold\" + Local Search]: ",runtimes.Burst)
    print("Our-SubBurst runtime [No \"Unfold\" + Local Search]: ",runtimes.SubBurst)

    # -----------------------------------
    #
    #      Compare _Centered Pixels_
    #
    # -----------------------------------

    # pix_hw = [1,1]
    # pix_hw = [h-1,w-1]
    pix_hw = [h//2,w//2]
    # pix_hw = [h//2+3,w//2-4]
    print("Our-L2-Local Output: ",vals.L2Local[:,pix_hw[0],pix_hw[1],:])
    print("Burst-L2-Local Output: ",vals.Burst[:,pix_hw[0],pix_hw[1],:])
    print("SubBurst-L2-Local Output: ",vals.SubBurst[:,pix_hw[0],pix_hw[1],:])

    vBurst = vals.Burst[:,pix_hw[0],pix_hw[1],:].cpu().numpy()[0]
    vSubBurst = vals.SubBurst[:,pix_hw[0],pix_hw[1],:].cpu().numpy()[0]
    iBurstValid = np.where(vBurst < 1e10)
    iSubBurstValid = np.where(vSubBurst < 1e10)
    assert np.all(iBurstValid[0] == iSubBurstValid[0]),"Equal OOB"
    # assert np.all(iBurstValid[1] == iSubBurstValid[1]),"Equal OOB"


    vBurst = vBurst[ iBurstValid ]
    vSubBurst = vSubBurst[ iSubBurstValid ]
    neqLocs = np.where(vBurst - vSubBurst)
    blockIndicesNeq = iBurstValid[0][neqLocs[0]]
    assert len(blockIndicesNeq) == 0, "Both burst types must be equal."
    # print("blockLabels")
    # print(blockLabels.shape)
    # print(blockLabels[:,blockIndicesNeq])
    # print(blockLabels[:,-5:])

    # print(vals.Burst[:,pix_hw[0],pix_hw[1],:])
    # print(locs.Burst[:,pix_hw[0],pix_hw[1],0])
    loc_top1 = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("-"*20)
    # print(vals.Burst[:,pix_hw[0]-1,pix_hw[1]-1,:])
    # print(vals.Burst[:,pix_hw[0]-1,pix_hw[1]+1,:])
    # print(vals.Burst[:,pix_hw[0]+1,pix_hw[1]-1,:])
    # print(vals.Burst[:,pix_hw[0]+1,pix_hw[1]+1,:])
    print("-"*20)

    output = []
    output.append(vals.Burst[:,pix_hw[0],pix_hw[1],:].cpu().numpy())
    # output.append(vals.Burst[:,pix_hw[0]-1,pix_hw[1]-1,:].cpu().numpy())
    # output.append(vals.Burst[:,pix_hw[0]-1,pix_hw[1]+1,:].cpu().numpy())
    # output.append(vals.Burst[:,pix_hw[0]+1,pix_hw[1]-1,:].cpu().numpy())
    # output.append(vals.Burst[:,pix_hw[0]+1,pix_hw[1]+1,:].cpu().numpy())
    output = np.array(output)[:,0]

    print(flow.shape,block.shape)
    gt_dist = compute_gt_burst(burstPad,pix_hw,block,ps,nblocks)
    top1_val = vals.Burst[0,pix_hw[0],pix_hw[1],0].item()
    top1_loc = locs.Burst[:,pix_hw[0],pix_hw[1],0].cpu().numpy()
    print("Burst Output @ Top1: ",top1_val)
    print("GT-Burst Output @ GT Flow: ",gt_dist)
    print("Burst Output @ Top1: ",top1_loc)
    print("GT Block",block)
    
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
    for i in range(10):
        test_burst_nnf_sample()
    
if __name__ == "__main__":
    test_burst_nnf()
