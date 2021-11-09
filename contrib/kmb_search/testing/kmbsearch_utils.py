
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
from datasets import get_dataset
from datasets.transforms import get_dynamic_transform

# -- faiss --
import faiss

# -- faiss/contrib --
from kmb_search.testing.utils import set_seed

def get_cfg_defaults():
    import settings

    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = 32

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 1
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 1
    cfg.dynamic_info.textured = True
    cfg.random_seed = 0

    return cfg

def get_image(std,t,h,w,seed,device):
    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.nframes = t
    cfg.random_seed = seed
    cfg.frame_size = [h,w]
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.noise_params.g.std = std
    nbatches = 20

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    print("load image dataset.")
    data,loaders = get_dataset(cfg,"dynamic")
    image_iter = iter(loaders.tr)    
    sample = next(image_iter)
    noisy = sample['dyn_noisy']
    clean = sample['dyn_clean']
    flows_gt = sample['ref_flow']
    print("flows_gt.shape: ",flows_gt.shape)

    # -- shaping --
    noisy = rearrange(noisy,'t 1 c h w -> c t h w')
    clean = rearrange(clean,'t 1 c h w -> c t h w')
    flows_gt = rearrange(flows_gt,'1 t h w two -> two t h w')

    # -- device --
    noisy = noisy.to(device)
    clean = clean.to(device)
    flows_gt = flows_gt.to(device)

    return noisy,clean,flows_gt

def get_rand_image(std,t,h,w,seed,device):

    # -- get dynamic xform --
    dynamic_info = edict()
    dynamic_info.mode = 'global'
    dynamic_info.nframes = t
    dynamic_info.ppf = 1
    dynamic_info.frame_size = [h,w]
    dyn_xform = get_dynamic_transform(dynamic_info,None)
    t = dynamic_info.nframes

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
    noisy = clean + torch.FloatTensor(noise).to(device)

    return noisy,clean,flows_gt

def kmbsearch_setup(k,t,h,w,c,ps,std,device,seed=123):

    # ---------------------------
    #
    #        Init Vars
    #
    # ---------------------------

    # -- set seed --
    set_seed(seed)

    # -- outputs --
    vals = edict()
    locs = edict()
    runtimes = edict()
    
    # ----------------------------
    # 
    #    Create Images & Flow
    # 
    # ----------------------------

    # -- get image --
    noisy,clean,flows_gt = get_image(std,t,h,w,seed,device)
    save_image("kmbs_noisy.png",noisy,bdim=1)

    # -- flow to blocks --
    # (dx,dy) -> (-dy,dx) with "y" [0,M] -> [M,0]
    blocks_gt = torch.stack([-flows_gt[1],flows_gt[0]],dim=0)

    # -- blocks to inds;  --
    # What is inds?: aligned[c,t,inds[0,i,j],inds[1,i,j]] = noisy[c,t,i,j]
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    print("coords.shape: ",coords.shape)
    inds_gt = blocks_gt + coords
    
    return noisy,clean,inds_gt,flows_gt,blocks_gt
    
