
# -- python imports --
import time
import torch
import torchvision
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
from sub_burst import evalAtLocs
from sub_burst import runBurstNnf as runSubBurstNnf
from bp_search import runBpSearch
from nnf_share import padAndTileBatch,padBurst,tileBurst,pix2locs,padLocs,warp_burst_from_pix


def run_test(burst,patchsize,nblocks):

    # -- 1.) run l2 local search --
    img_shape = list(burst.shape[-3:])
    device = burst.device
    vals,pix = nnf_utils.runNnfBurst(burst, patchsize, nblocks,
                                      k=1, valMean = 0,
                                      img_shape = None)
    vals = torch.mean(vals,dim=0).to(device)
    l2_vals = vals
    print("pix.shape ",pix.shape)
    pix = pix.to(device)
    l2_locs = pix2locs(pix)
    print("l2_locs.shape ",l2_locs.shape)
    locs = pix2locs(pix)
    
    print(locs[:,0,15,16,:])
    print(locs[:,0,16,15,:])
    print(locs[:,0,16,16,:])
    print("locs.shape: ",locs.shape)


    # -- 3.) warp burst to top location --
    print("burst.shape ",burst.shape)
    wburst = padAndTileBatch(burst,patchsize,nblocks)
    print("wburst.shape ",wburst.shape)
    pixPad = (wburst.shape[-1] - burst.shape[-1])//2
    ppix = padLocs(pix+pixPad,pixPad)
    print("ppix.shape ",ppix.shape)
    warped_burst = warp_burst_from_pix(wburst,ppix,nblocks)
    # warped_burst = center_crop(warped_burst,pshape)
    print("warped_burst.shape ",warped_burst.shape)
    img_shape[0] = wburst.shape[-3]
    # print(ppix[0,0,16+1,16+1,0])
    # print(burst[0,0,0,16,16])
    # ppad = 1
    # print(wburst[0,0,-5:,16+ppad,16+ppad])
    # ppad = 2
    # print(warped_burst[0,0,0,0,16+ppad,16+ppad])
    # print(warped_burst[0,0,0,0,15+ppad,16+ppad])
    # print(warped_burst[0,0,0,-5:,16+ppad,15+ppad])
    # print(torch.sum(torch.abs(wburst[0,0,:,16+1,16+1] - warped_burst[0,0,0,:,16+ppad,15+ppad])))
    # print(warped_burst[0,0,0,0,15+ppad,15+ppad])
    # ppad = 1
    # print(warped_burst[0,0,0,0,16+ppad,16+ppad])
    # print(warped_burst[0,0,0,0,15+ppad,16+ppad])
    # print(warped_burst[0,0,0,0,16+ppad,15+ppad])
    # print(warped_burst[0,0,0,0,15+ppad,15+ppad])

    # -------------------------------
    #
    #   Correctly set values using  
    #   Sub-Ave instead of L2
    #
    # -------------------------------
    print(l2_locs.shape)
    print(type(locs))
    # l2_locs[:,:,0,0] = l2_locs[:,:,16,16]

    # print("[BpSearch]: burst.shape ",burst.shape)
    sub_vals,sub_locs = runSubBurstNnf(burst,patchsize,
                                   nblocks,k=1, # patchsize=1 since tiled
                                   blockLabels=None)
    print("e_locs.shape: ",sub_locs.shape)
    sub_locs_rs = rearrange(sub_locs,'i t h w k two -> t i h w k two')

    # ----------------------------
    # -- reassign for testing! --
    # ----------------------------
    in_eval_vals = l2_vals
    # in_eval_locs_rs = l2_locs
    in_eval_locs_rs = sub_locs_rs


    print("running eval at locs.")
    print("in_eval_locs_rs.shape ",in_eval_locs_rs.shape)
    psHalf = patchsize//2
    in_eval_locs_rs_pad = padLocs(in_eval_locs_rs,psHalf,mode='extend')
    print("in_eval_locs_rs.shape ",in_eval_locs_rs_pad.shape)
    print("burst.shape ",burst.shape)
    # e_vals,e_locs = evalAtLocs(burst, in_eval_locs_rs_pad,
    #                            patchsize, nblocks,
    #                            img_shape=None)
    e_vals,e_locs = evalAtLocs(wburst,
                               in_eval_locs_rs, 1,
                               nblocks,
                               img_shape=img_shape)

    # e_locs = e_locs[0]
    print("e_locs.shape: ",e_locs.shape)
    print("complete.")
    # e_vals,e_locs = evalAtLocs(wburst,
    #                            l2_locs, 1,
    #                            nblocks,
    #                            img_shape=img_shape)
    # e_vals,e_locs = runSubBurstNnf(burst,patchsize,
    #                                nblocks,k=1, # patchsize=1 since tiled
    #                                blockLabels=None)
    e_locs_rs = rearrange(e_locs,'i t h w k two -> t i h w k two')

    print("-"*30)
    print("Comparing Outputs")
    print("-"*30)
    
    pix_list = [[0,0],[1,0],[16,16]]
    for pix_hw in pix_list:
        i,j = pix_hw
        print(f"------ Locs ({i},{j}) ------")
        delta_sub_l2 = torch.sum(torch.abs(sub_locs_rs[:,:,i,j] - l2_locs[:,:,i,j]))
        print("[Delta (SubBurst - L2)]: ",delta_sub_l2)
        delta_sub_eval = torch.sum(torch.abs(sub_locs_rs[:,:,i,j] - e_locs_rs[:,:,i,j]))
        print("[Delta (SubBurst - Eval)]: ",delta_sub_eval)
        # print("SubBurst: ",sub_locs_rs[:,:,i,j])
        # print("Eval: ",e_locs_rs[:,:,i,j])
        # print("L2: ",l2_locs[:,:,i,j])
    
        print(f"------ Vals ({i},{j}) ------")
    
        print("SubBurst: ",sub_vals[:,i,j])
        print("Eval: ",e_vals[:,i,j])
        print("L2: ",l2_vals[:,i,j])
        print("-"*50)
    
    print("-"*30)

    print("SUCCESS!")



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def test_eval_at_flow():

    # -- settings --
    seed = 234
    seed = 345
    set_seed(seed)

    # -- image params --
    h,w,c = 32,32,3
    patchsize = 11
    nblocks = 3
    k = 3
    gpuid = 0

    # -- derived variables --
    ps = patchsize
    pad = int(ps//2)+int(nblocks//2)
    pads = (pad,pad,pad,pad)
    
    # -- apply dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = 4
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
    noise = np.random.normal(loc=0,scale=0.19,size=(t,1,c,h,w)).astype(np.float32)
    burst += torch.FloatTensor(noise).to(device)
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
    print("original burst ",burst.shape)
    print("pad no tile ",burstPad.shape)

    # -- get block labels --
    blockLabels,_ = bnnf_utils.getBlockLabels(None,nblocks,np.float32,
                                              'cpu',False,nframes)


    # ----------------------------
    # 
    #    Execute Test
    # 
    # ----------------------------

    run_test(burst,patchsize,nblocks)



if __name__ == "__main__":
    test_eval_at_flow()
