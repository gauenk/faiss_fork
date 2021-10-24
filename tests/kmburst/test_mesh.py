
    
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
from bp_search import runBpSearch,create_mesh_from_ranges
from nnf_share import padAndTileBatch,padBurst,tileBurst,pix2locs
from kmb_search import runKmSearch,init_zero_locs,init_zero_traj,jitter_traj_ranges

# -- local imports --
sys.path.append("/home/gauenk/Documents/faiss/tests/")
from kmburst.utils import compute_gt_burst,set_seed

def exp_setup():
    # seed = 234
    seed = 345
    # seed = 456
    # seed = 678
    set_seed(seed)

    
    # h,w,c = 1024,1024,3
    # h,w,c = 32,32,3
    # h,w,c = 16,16,3
    # h,w,c = 17,17,3
    # h,w,c = 512,512,3
    # h,w,c = 256,256,3
    # h,w,c = 128,128,3
    h,w,c = 64,64,2
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

def create_mesh():

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
    std = 20
    noise = np.random.normal(loc=0,scale=std/255.,size=(t,1,c,h,w)).astype(np.float32)
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
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(burst, patchsize,
                                              nblocks, 1,
                                              valMean = valMean,
                                              blockLabels=None)
    runtimes.L2Local = time.perf_counter() - start_time
    vals.L2Local = nnf_vals[:,0]
    locs.L2Local = pix2locs(nnf_locs)[:,0]

    # ------------------------------------------
    #
    #     Compute FAISS KmBurstNnf
    #
    # ------------------------------------------

    print("-"*30)
    print("KMeans Burst")
    print("-"*30)
    gt_dist = 0.
    start_time = time.perf_counter()
    _vals,_locs = runKmSearch(burst, patchsize,nblocks, k = k,
                              std = std,search_space=None, ref=None)
    runtimes.KmBurst = time.perf_counter() - start_time
    print("_vals.shape: ",_vals.shape)
    print("_locs.shape: ",_locs.shape)
    # vals.shape: (i,k,h,w)
    # locs.shape: (i,k,t,h,w,2)
    vals.KmBurst = _vals[0,0]
    locs.KmBurst = _locs[0,0]
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

    # -- return stuff --
    sizes = [nframes,nimages,h,w]

    return sizes 

def check_rollout(sizes):
    # check_rollout_v1()
    # check_rollout_v2(sizes)
    check_rollout_v3(sizes)

def check_rollout_v1():
    s,t,h,w = 9,3,8,8
    inputs = torch.arange(s*t*h*w*2).reshape(2,t,s,h,w)
    outputs = []
    for ti in range(t):
        for si in range(s):
            for hi in range(h):
                for wi in range(w):
                    for bi in range(2):
                        outputs.append(inputs[bi,ti,si,hi,wi].item())
    outputs = torch.LongTensor(outputs)
    outputs = rearrange(outputs,'(t s h w two) -> two t s h w',s=s,t=t,h=h,w=w)
    delta = torch.sum(torch.abs(inputs - outputs)).item()
    print("rollout delta: ",delta)
    # assert delta < 1e-10, "delta should be zero."

def check_rollout_v2(sizes):
    sranges = read_search_ranges(sizes)
    print("sranges.shape: ",sranges.shape)
    print_unique_hw(sranges)

def check_rollout_v3(sizes):
    mesh = read_mesh_text(sizes)
    print("mesh.shape: ",mesh.shape)
    print_unique_hw(mesh)


def read_tensor5d(fn,sizes):
    t,i,h,w = sizes 
    h,w = 16,16
    tensor5d = np.loadtxt(fn,dtype=int,delimiter=',')
    # tensor5d = repeat(tensor5d,'(t s h w two) -> 1 i two t s h w',i=i,two=2,h=h,w=w,t=t)
    tensor5d = repeat(tensor5d,'(t s h w two) -> 1 i two t s h w',i=i,two=2,h=h,w=w,t=t)
    return tensor5d

def read_mesh_text(sizes):
    fn = "blocks.txt"
    return read_tensor5d(fn,sizes)

def read_search_ranges(sizes):
    fn = "search_ranges.txt"
    return read_tensor5d(fn,sizes)

def print_unique_hw(mesh):
    t,s,h,w = mesh.shape[-4:]
    for hi in range(h):
        for wi in range(w):
            for ti in range(t):
                mesh_2d = rearrange(mesh[...,ti,:,hi,wi],'1 i two s -> (1 i s) two')
                uniq = unique_mesh(mesh_2d)
                print(ti,hi,wi,uniq)
    mesh_2d = rearrange(mesh,'1 i two t s h w -> (1 i s t h w) two')
    uniq = unique_mesh(mesh_2d)
    print("Overall: ",uniq)

def unique_mesh(mesh_2d):
    smesh = ['%d_%d' % (m[0],m[1]) for m in mesh_2d]
    uniques = np.unique(smesh)
    return uniques

def check_mesh(sizes):

    # -- create ground-truth ranges --
    nframes,nimages,h,w = sizes 
    traj = init_zero_traj(nframes,nimages,h,w)
    sranges = jitter_traj_ranges(traj,3)

    # -- create ground-truth mesh --
    ref = nframes//2
    sranges_fmt = rearrange(sranges,'1 i two t r2 h w -> r2 i h w t two')
    sranges_fmt = sranges_fmt.to(0)
    mesh_gt = create_mesh_from_ranges(sranges_fmt,ref).cpu().numpy()
    mesh_gt = rearrange(mesh_gt,'s i h w t two -> 1 i two t s h w')
    mesh_gt = mesh_gt[...,-16:,-16:]
    print("mesh_gt.shape: ",mesh_gt.shape)
    print_unique_hw(mesh_gt)

    # -- read in saved mesh --
    mesh_tgt = read_mesh_text(sizes)
    print("mesh_tgt.shape: ",mesh_tgt.shape)
    print_unique_hw(mesh_tgt)

    # -- compute the difference --
    delta = np.sum(np.abs(mesh_gt - mesh_tgt)).item()
    print("Delta: ",delta)

    assert delta < 1e-8,"delta should be zero."

def test_mesh():
    # sizes = create_mesh()
    sizes = [4,1,64,64]
    # check_rollout(sizes)
    check_mesh(sizes)

if __name__ == "__main__":
    test_mesh()
