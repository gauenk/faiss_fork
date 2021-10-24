

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
from bp_search import runBpSearch
from nnf_share import padAndTileBatch,padBurst,tileBurst,pix2locs,warp_burst_from_locs,locs2flow

center_crop = torchvision.transforms.functional.center_crop


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_test_warp_burst():
    # -- settings --
    seed = 234
    seed = 345
    set_seed(seed)

    h,w,c = 32,32,3
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    # ps,nblocks = 11,10
    patchsize = 5
    nblocks = 5
    k = 3#(nblocks**2)**2
    gpuid = 0

    # -- derived variables --
    ps = patchsize
    pad = int(ps//2)+int(nblocks//2)
    pads = (pad,pad,pad,pad)
    
    # -- apply dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = 4
    dynamic_info.ppf = 2
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
    clean = dyn_result[0][:,None].to(gpuid)
    flow =  dyn_result[-2]
    device = clean.device

    # -- format data --
    noise = np.random.normal(loc=0,scale=0.19,size=(t,1,c,h,w)).astype(np.float32)
    noisy = clean + torch.FloatTensor(noise).to(device)
    block = np.c_[-flow[:,1],flow[:,0]] # (dx,dy) -> (dy,dx) with "y" [0,M] -> [M,0]
    nframes,nimages,c,h,w = clean.shape
    isize = edict({'h':h,'w':w})
    flows = repeat(flow,'t two -> 1 p t two',p=h*w)
    assert dynamic_info.ppf <= (nblocks//2), "nblocks is too small for ppf"

    # -- add padding --
    t,b,c,h,w = clean.shape
    cleanPad,offset = bnnf_utils.padBurst(clean[:,0],(c,h,w),ps,nblocks)[:,None],2
    print("original clean ",clean.shape)
    print("pad no tile ",cleanPad.shape)

    # -- get block labels --
    blockLabels,_ = bnnf_utils.getBlockLabels(None,nblocks,np.float32,
                                              'cpu',False,nframes)
    
    # ------------------------------------------
    #
    #     Compute FAISS SubBurstNnf
    #
    # ------------------------------------------

    print("-"*30)
    print("Sub Burst")
    print("-"*30)
    gt_dist = 0.
    valMean = gt_dist
    start_time = time.perf_counter()
    print("clean.shape: ",clean.shape)
    _vals,_locs = sbnnf_utils.runBurstNnf(clean, patchsize,
                                          nblocks, k = k,
                                          valMean = valMean,
                                          blockLabels=None,
                                          ref=None,
                                          img_shape=clean.shape[-3:])
    runtimes.SubBurst = time.perf_counter() - start_time
    vals.SubBurst = _vals[0][None,:] # include nframes 
    # locs.SubBurst = _locs[:,0]
    locs.SubBurst = _locs[0]


    locs = locs.SubBurst
    print("locs.shape ",locs.shape)
    isize_e = edict({'h':h,'w':w})
    isize_l = [h-nblocks//2-patchsize//2,w-nblocks//2-patchsize//2]
    locs = locs[:,None]
    print(locs.shape)
    ref = nframes//2


    print(locs[:,:,16,16,0])
    flows_rs = rearrange(flows,'i (h w) t two -> i h w t two',h=h,w=w)
    print(flows_rs[:,16,16,:])
    flow_xfer = locs2flow(locs)
    print(flow_xfer.shape)
    print(flow_xfer[:,:,16,16,0])

    cc_clean = center_crop(clean,isize_l)
    warped = warp_burst_from_locs(clean,locs,patchsize,isize)[0]
    warped = center_crop(warped,isize_l)
    delta = torch.sum(torch.abs(cc_clean[[ref]] - warped))
    # delta = torch.sum(torch.abs(warped[[ref]] - warped))
    print(delta)
    
    cc_clean = cc_clean.to('cpu')
    aligned = align_from_flow(clean,flows,patchsize,isize=isize)
    aligned = center_crop(aligned,isize_l)
    delta = torch.sum(torch.abs(cc_clean[[ref]] - aligned))
    # delta = torch.sum(torch.abs(aligned[[ref]] - aligned))
    print(delta)


if __name__ == "__main__":
    run_test_warp_burst()
