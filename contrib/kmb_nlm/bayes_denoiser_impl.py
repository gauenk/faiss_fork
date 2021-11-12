
# -- python --
import torch
import numpy as np

# -- numba --
from numba import jit,njit,prange


def compute_bayes_denoised_frames(burst,inds,means,covs,std):

    # -- shape -- 
    device = burst.device
    c,t,h,w = burst.shape

    # -- denoiser output --
    denoised = np.zeros((c,h,w))

    # -- inv covs --
    inv_covs = compute_inv_covs(covs,std)

    # -- to numpy --
    burst_nba = burst.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    means_nba = means.cpu().numpy()
    covs_nba = covs.cpu().numpy()
    inv_covs_nba = inv_covs.cpu().numpy()

    # -- exec numba --
    compute_bayes_denoised_frames_numba(denoised,burst_nba,inds_nba,
                                        means_nba,covs_nba,inv_covs_nba)

    # -- to tensor --
    denoised = torch.FloatTensor(denoised).to(device)

    return denoised

def compute_inv_covs(covs,std):
    """ i know this is not actually the _right_ way to do this """
    f,f,h,w = covs.shape
    covs = rearrange(covs,'f f h w -> (h w) f f')
    inv_covs = torch.linalg.inv(covs)
    inv_covs = rearrange(inv_covs,'(h w) f f -> f f h w',h=h,w=w)
    return inv_covs

@njit
def compute_bayes_denoised_frames_numba(denoised,burst,inds,means,covs,inv_covs):

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- shapes --
    c,t,h,w = burst.shape
    f,h,w = means.shape

    # -- helpful vars --
    nframes = t
    ref = t//2

    # -- exec --
    for hi in prange(h):
        for wi in prange(w):
            for ti in range(t):
                
                blkH = inds[0,ti,hi,wi]
                blkW = inds[1,ti,hi,wi]
                top,left = blkH-psHalf,blkW-psHalf

                est = np.zeros(f)
                for fi in range(f):
                    for fj in range(f):

                        q_i = 1.
                        coeff = covs[fi,fj,hi,wi] * inv_covs[fi,fj,hi,wi]
                        est[fi] += coeff * (q_i - means)

                denoised[ci,hi,wi] = means[ci,hi,wi] + est
