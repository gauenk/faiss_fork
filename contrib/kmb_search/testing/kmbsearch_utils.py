
# -- python imports --
import torch
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat
from PIL import Image

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from pyutils import save_image,get_img_coords
from datasets.transforms import get_dynamic_transform

# -- faiss --
import faiss

# -- faiss/contrib --
from kmb_search.testing.utils import set_seed

def kmbsearch_setup(k,t,h,w,c,ps,std,device,seed=123):

    # ---------------------------
    #
    #        Init Vars
    #
    # ---------------------------

    # -- set seed --
    set_seed(seed)

    # -- apply dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = t
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
    # image[1::2,::2],image[::2,1::2] = 1.,1.
    image = np.uint8(image*255.)
    imgPIL = Image.fromarray(image)
    dyn_result = dyn_xform(imgPIL)

    # -- format outputs --
    clean = dyn_result[0][:,None].to(device)
    flow =  torch.IntTensor(dyn_result[-2]).to(device)
    flows_gt = repeat(flow,'t two -> two t h w',h=h,w=w)

    # -- format burst data --
    clean = rearrange(clean,'t 1 c h w -> c t h w')
    noise = np.random.normal(loc=0,scale=std/255.,size=(c,t,h,w)).astype(np.float32)
    burst = clean + torch.FloatTensor(noise).to(device)
    save_image("burst.png",burst,bdim=1)

    # -- flow to blocks --
    # (dx,dy) -> (-dy,dx) with "y" [0,M] -> [M,0]
    blocks_gt = torch.stack([-flows_gt[1],flows_gt[0]],dim=0)

    # -- blocks to inds;  --
    # What is inds?: aligned[c,t,inds[0,i,j],inds[1,i,j]] = burst[c,t,i,j]
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    print("coords.shape: ",coords.shape)
    inds_gt = blocks_gt + coords
    
    return burst,clean,inds_gt,flows_gt,blocks_gt
    
