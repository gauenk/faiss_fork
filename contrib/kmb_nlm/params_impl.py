
# -- python --
import torch
import numpy as np

# -- numba --
from numba import jit,njit,prange

def compute_nlm_params(burst,inds,modes,ps):

    # -- create weights --
    device = burst.device
    c,t,h,w = burst.shape

    # -- init output --
    f = c*ps*ps
    means = np.zeros((f,h,w))
    covs = np.zeros((f,f,h,w))

    # -- to numpy --
    burst_nba = burst.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    modes_nba = modes.cpu().numpy()

    # -- run numba --
    compute_nlm_params_numba(means,covs,burst_nba,inds_nba,modes_nba,ps)

    # -- to tensor --
    means = torch.FloatTensor(means).to(device)
    covs = torch.FloatTensor(covs).to(device)

    return means,covs

@njit
def compute_nlm_params_numba(means,covs,burst,inds,modes,ps):

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- shapes --
    c,t,h,w = burst.shape
    t,h,w = weights.shape

    # -- helpful vars --
    nframes = t
    ref = t//2
    psHalf = ps//2

    # -- exec --
    for hi in prange(h):
        for wi in prange(w):
            for ti in prange(t):
                
                blkHref = inds[0,ref,hi,wi]
                blkWref = inds[1,ref,hi,wi]
                topRef,leftRef = blkHref-psHalf,blkWref-psHalf

                blkH = inds[0,ti,hi,wi]
                blkW = inds[1,ti,hi,wi]
                top,left = blkH-psHalf,blkW-psHalf

                weight = 0
                for ci in range(c):
                    for pi in range(ps):
                        for pj in range(ps):

                            
                            bH = bounds(top+pi,h-1)
                            bW = bounds(left+pj,w-1)
                            b = burst[ci,ti,bH,bW]

                            bH_ref = bounds(topRef+pi,h-1)
                            bW_ref = bounds(leftRef+pj,w-1)
                            b_ref = burst[ci,ref,bH_ref,bW_ref]

                            weight += ((b - b_ref)**2 - modes[ti,hi,wi])/(nframes-1)
                weight = np.exp(-weight)
                weights[ti][hi][wi] = weight
