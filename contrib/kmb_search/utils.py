
# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

def get_ref_centroids(clusters,centroids,inds,ref=None):
    t = clusters.shape[0]
    c,tK,s,h,w,ps,ps = centroids.shape
    if ref is None: ref = t//2
    inds = clusters[ref].type(torch.long)
    inds = repeat(inds,'s h w -> c 1 s h w p1 p2',c=c,p1=ps,p2=ps)
    ref_centroids = centroids.gather(dim=1,index=inds)[:,0]
    return ref_centroids

def get_optional_field(pydict,field,default):
    if pydict is None: return default
    if not(isinstance(pydict,dict)): raise TypeError("pydict must be a dict.")
    if not(field in pydict): return default
    return pydict[field]

def get_gt_info(gt_info):
    if not(gt_info is None):
        if "clean" in gt_info:
            clean = gt_info['clean']
        if "indices" in gt_info:
            indices_gt = gt_info['indices']
    else:
        clean = None
        indices_gt = None
    return clean,indices_gt

def tiled_search_frames(nframes,nfsearch,nsiters,ref):
    assert nfsearch <= nframes,"must search leq nframes"
    sframes = torch.zeros(nsiters,nfsearch)

    # -- create first cycle in [0,nfsearch-2] --
    m = 0
    base = torch.arange(nfsearch-1).type(torch.int)
    for i in range(nsiters):
        sframes[i,:-1] = (base + m) % nframes
        # print(sframes[i,:],torch.where(ref == sframes[i,:])[0],m)
        # if len(torch.where(ref == sframes[i,:])[0]) == 0:
        #        sframes[i,0] = ref
        # sframes[i,:] = sframes[i,:] % nframes
        if m == (nframes-nfsearch): m = 0
        else: m = m+1

    # -- add value if greater than reference --
    sframes[np.where(sframes >= ref)] += 1

    # -- fill last with ref --
    sframes[:,-1] = ref

    # -- sort each row --
    sframes = torch.sort(sframes,dim=1)[0]

    return sframes

def mesh_from_ranges(search_ranges,search_frames,curr_blocks,ref):
    # -- create mesh of current blocks --
    search_frames = search_frames.type(torch.long) # for torch indexing
    two,t,s,h,w = search_ranges.shape
    sranges = rearrange(search_ranges,'two t s h w -> t s 1 h w two')
    sranges = sranges[search_frames]
    sranges = rearrange(sranges,'t s 1 h w two -> s 1 h w t two')
    m_ref = torch.where(search_frames == ref)[0][0].item()
    mesh = create_mesh_from_ranges(sranges,m_ref)
    mesh = rearrange(mesh,'b 1 h w g two -> two g b h w')

    # -- append the current state frames --
    nblocks = mesh.shape[-3]
    blocks = curr_blocks.clone()
    blocks = repeat(blocks,'two t h w -> two t b h w',b=nblocks)
    for group,frame in enumerate(search_frames):
        # if frame == ref: continue
        # if group == m_ref: continue
        # if group == ref: continue
        blocks[:,frame] = mesh[:,group]
    
    # print("-"*30)
    # print("ref, nblocks: ",ref,nblocks)
    # print("mesh.shape: ",mesh.shape)
    # print("blocks.shape: ",blocks.shape)
    # # print(mesh[:,:,:,4,5].transpose(1,2).transpose(0,1))
    # for s in range(blocks.shape[2]):
    #     print(blocks[:,:,[s],4,5].transpose(1,2).transpose(0,1).cpu().numpy())
    # print(search_ranges.shape)
    # exit()

    return blocks

def jitter_search_ranges(nrange,t,h,w,ref=None,offset=True):

    # -- create ranges --
    if ref is None: ref = t//2
    mrange = nrange//2
    sranges = torch.zeros(nrange,nrange,2)
    for i in range(nrange):
        for j in range(nrange):
            sranges[i,j,0] = i - mrange
            sranges[i,j,1] = j - mrange
    sranges = rearrange(sranges,'r1 r2 two -> (r1 r2) two')
    
    # -- repeat to full shape --
    sranges = repeat(sranges,'r2 two -> two t r2 h w',t=t,h=h,w=w)
    sranges = sranges.type(torch.int).contiguous()

    # -- zero out ref ranges --
    sranges[:,ref] = torch.zeros_like(sranges[:,0])

    # -- from relative to absolute coordinates --
    if offset:
        r2 = nrange**2
        coords = get_img_coords(t,r2,h,w)
        sranges = sranges + coords

    return sranges

def jitter_traj_ranges(trajs,jsize,ref=None,offset=True):
    k,i,two,t,h,w = trajs.shape
    jitter = jitter_search_ranges(jsize,t,h,w,ref,offset).to(trajs.device)
    jitter = repeat(jitter,'two t r2 h w -> k i two t r2 h w',k=k,i=i)
    trajs = repeat(trajs,'k i two t h w -> k i two t r2 h w',r2=jsize**2)
    jtrajs = trajs + jitter
    return jtrajs

def init_zero_traj(nframes,nimages,h,w):
    locs = init_zero_locs(nframes,nimages,h,w)
    trajs = rearrange(locs,'t i h w k two -> k i two t h w')        
    return trajs

def init_zero_locs(nframes,nimages,h,w):
    locs = torch.zeros(nframes,nimages,h,w,1,2)
    locs = locs.type(torch.int)
    return locs

def compute_l2_mode(std,patchsize):
    return 0.

def divUp(a,b): return (a-1)//b+1

def initialize_indices(coords,search_ranges,indices_gt):


    #
    # -- use the coords --
    #

    # vprint(rinit.shape,search_ranges.shape)
    # curr_indices = torch.zeros_like(search_ranges)
    curr_indices = search_ranges[:,:,0]
    # curr_indices = coords.clone()
    # vprint("curr_indices.shape: ",curr_indices.shape)
    # coords.clone() --> (2,t,h,w)

    #
    # -- random inits --
    #

    # rinit = search_ranges.shape[2]//2
    # rmax = search_ranges.shape[2]
    # rinit = torch.randint(0,rmax,(t,1,h,w)).to(device)

    #
    # -- search ranges --
    #

    # curr_indices[0] = torch.gather(search_ranges[0],dim=1,index=rinit)
    # curr_indices[1] = torch.gather(search_ranges[1],dim=1,index=rinit)
    # curr_indices = curr_indices.type(torch.long)
    # curr_indices = curr_indices[:,:,0]
    # curr_indices = indices_gt.clone()

    return curr_indices

def pick_fill_frames(sframes,nfsearch,t,alpha,s_iter,device):
    
    # -- simulate rand nums --
    # alpha = 5-2./(math.log10(s_iter+1)+1)
    alpha = 5-2./(s_iter+1.)
    # alpha = 2
    beta = torch.distributions.beta.Beta(alpha,1)
    rand = beta.sample().item()
    # print(alpha,rand)
    # rand = torch.rand(1).item()
    # rand = int(rand*(t-nfsearch-1))
    rand = int(rand*(t-nfsearch-1))
    fgrid = np.arange(t)
    sframes_np = sframes.cpu().numpy()
    sframes_set = set(list(sframes_np))
    sframes_not = set(list(fgrid))
    sframes_not = sframes_not - sframes_set
    sframes_not = np.sort(list(sframes_not))
    fgrid = np.random.choice(sframes_not,rand,replace=False)
    fgrid = np.sort(np.concatenate([fgrid,sframes_np]))

    # -- fill with all frames after time --
    # coin_flip = torch.rand(1).item()
    # if s_iter > 20 and coin_flip > 0.5:
    #     fgrid = torch.arange(nframes).to(device)


    # -- to torch --
    fgrid = torch.LongTensor(fgrid).to(device)

    return fgrid
        

