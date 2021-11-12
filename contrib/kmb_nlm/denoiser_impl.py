

# -- python --
import torch
import numpy as np

# -- numba --
from numba import jit,njit,prange

def eigen_denoiser(eigVecs,eigVals):
    mm = torch.matmul
    U = eigVecs
    Ut = U.transpose(1,2)
    W = torch.diag_embed(eigVals)
    UW = mm(U,W)
    # diag = torch.arange(U.shape[-1])
    # augEigs = torch.zeros(U.shape[0],U.shape[-1]).to(device)
    # augEigs[:,:rank] = bEigs
    # UW[:,diag,diag] = augEigs
    # torch.diag(UW) = torch.diag(UW) * augEig
    # UW = torch.diag(U)
    # print(bEigs.shape,U.shape,UW.shape,noisy.shape)
    dframes = noisy_ave + mm(UW,mm(Ut,noisy))
    dref = torch.mean(dframes,dim=-1)
    return dframes,dref

def compute_denoised_frames(burst,inds,weights):

    # -- shape -- 
    device = burst.device
    c,t,h,w = burst.shape

    # -- denoiser output --
    denoised = np.zeros((c,h,w))

    # -- to numpy --
    burst_nba = burst.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    weights_nba = weights.cpu().numpy()

    # -- exec numba --
    compute_denoised_frames_numba(denoised,burst_nba,inds_nba,weights_nba)

    # -- to tensor --
    denoised = torch.FloatTensor(denoised).to(device)

    return denoised

@njit
def compute_denoised_frames_numba(denoised,burst,inds,weights):

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

    # -- exec --
    for hi in prange(h):
        for wi in prange(w):
            wsum = 0
            for ti in range(t):
                
                weight = weights[ti,hi,wi]
                blkH = inds[0,ti,hi,wi]
                blkW = inds[1,ti,hi,wi]

                for ci in range(c):
                    b = burst[ci,ti,blkH,blkW]
                    denoised[ci,hi,wi] += weight * b
                wsum += weight
            for ci in range(c):
                denoised[ci,hi,wi] /= wsum

