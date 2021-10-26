
# -- python imports --
import torch
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat
from PIL import Image

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from pyutils import save_image
from datasets.transforms import get_dynamic_transform

# -- faiss --
import faiss
PWD_TYPE = faiss.PairwiseDistanceCase
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges

# -- faiss/contrib --
from kmb_search.testing.utils import set_seed

def pwd_setup(k,t,h,w,c,ps,std,device):

    # ---------------------------
    #
    #        Init Vars
    #
    # ---------------------------

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
    burst = dyn_result[0][:,None].to(device)
    flow =  dyn_result[-2]

    # -- format data --
    noise = np.random.normal(loc=0,scale=std,size=(t,1,c,h,w)).astype(np.float32)
    burst += torch.FloatTensor(noise).to(device)
    block_gt = np.c_[-flow[:,1],flow[:,0]] # (dx,dy) -> (dy,dx) with "y" [0,M] -> [M,0]
    block_gt = torch.IntTensor(block_gt)
    save_image("burst.png",burst)
    nframes,nimages,c,h,w = burst.shape
    isize = edict({'h':h,'w':w})
    flows = repeat(flow,'t two -> 1 p t two',p=h*w)

    # -- to device --
    burst = rearrange(burst.to(device),'t 1 c h w -> c t h w')
    block_gt = block_gt.to(device)
    block_gt = repeat(block_gt,'t two -> two t h w',h=h,w=w)
    
    return burst,block_gt
    
