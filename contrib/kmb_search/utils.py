
import torch
from einops import rearrange,repeat


def jitter_search_ranges(nrange,t,h,w):

    # -- create ranges --
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
    return sranges

def jitter_traj_ranges(trajs,jsize):
    k,i,two,t,h,w = trajs.shape
    jitter = jitter_search_ranges(jsize,t,h,w).to(trajs.device)
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


