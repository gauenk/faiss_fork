# -- python --
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- pytorch --
import torch

# -- interface --
def compute_mode(*args,**kwargs):
    if kwargs['type'] == 'pairs':
        return compute_mode_pairs(*args)
    elif kwargs['type'] == 'burst':
        return compute_mode_burst(*args)
    elif kwargs['type'] == 'centroids':
        return compute_mode_centroids(*args)
    else:
        cmode = kwargs['type']
        raise KeyError(f"Unknown type of compute mode [{cmode}]")

def compute_mode_pairs(std,c,ps):
    P = ps * ps * c
    var = std**2
    p_ratio = ((P-2)/P)
    mode = p_ratio * 2 * var
    return mode

def compute_mode_burst(std,c,ps,t):
    P = ps * ps * c
    var = std**2
    p_ratio = ((P-2)/P)
    t_ratio = ( (t-1)/t )**2 + (t-1)/t**2
    mode = p_ratio * t_ratio * var
    return mode

def compute_mode_centroids(std,c,ps,sizes):

    # -- to numpy --
    device = sizes.device
    tK,s,h,w = sizes.shape
    sizes = sizes.cpu().numpy()
    P = ps*ps*c

    # -- exec on numba --
    modes4d = np.zeros_like(sizes).astype(np.float)
    modes3d = np.zeros((s,h,w)).astype(np.float)
    compute_mode_centroids_numba(std,P,sizes,modes4d,modes3d)

    # -- to torch --
    modes4d = torch.FloatTensor(modes4d)
    modes4d = modes4d.to(device)
    modes3d = torch.FloatTensor(modes3d)
    modes3d = modes3d.to(device)

    return modes4d,modes3d

@njit
def compute_mode_centroids_numba(std,P,sizes,modes4d,modes3d):
    tK,s,h,w = sizes.shape
    p_ratio = (P - 2) / P
    var = std**2
    for wi in prange(w):
        for hi in prange(h):
            for si in prange(s):
                for t0 in prange(tK):
                    if sizes[t0,si,hi,wi] == 0:
                        modes4d[t0,si,hi,wi] = 0
                        modes3d[si,hi,wi] = 0
                        continue
                    mode = 0
                    svar = 0
                    nsum = 0
                    for t1 in range(tK):
                        size = sizes[t1,si,hi,wi]
                        if size > 0: nsum += 1
                        if t1 == t0: continue
                        if size > 0: svar += var / size
                    size_t0 = sizes[t0,si,hi,wi]
                    var_t0 = var / size_t0
                    if nsum == 0:
                        svar,c2_t0 = 0,0
                    else:
                        svar = svar / nsum**2
                        c2_t0 = ( (nsum-1) / nsum )**2 * var_t0 + svar
                    modes4d[t0,si,hi,wi] = p_ratio * c2_t0
                modes3d[si,hi,wi] = p_ratio * c2_t0
                    
                    
