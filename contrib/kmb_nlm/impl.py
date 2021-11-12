"""
Some non-local means denoising method.
These methods will improve if the image registration quality is higher.

"""

# -- python --
import torch
import numpy as np
from scipy import ndimage
from einops import rearrange,repeat

# -- project imports --
from pyutils import save_image,get_img_coords

# -- import components --
from .eigs import mod_eigs
from .weights_impl import compute_nlm_weights
# not the same "params" as in inputs... should be named "weights"
# from .params_impl import compute_nlm_params 
from .denoiser_impl import compute_denoised_frames,eigen_denoiser
from .bayes_denoiser_impl import compute_bayes_denoised_frames
from ._align_interface import get_align_method
from .utils import return_optional_params,expanded_to_burst,expand_burst

def run_nlm(burst,ps,std,align_method="kmb",gt_info=None,params=None):

    # -- unpack --
    device = burst.device
    c,t,h,w = burst.shape
    nframes = t
    var = std**2
    psHalf = ps//2

    # -- get sim search method --
    align_fxn = get_align_method(align_method)

    # -- params --
    ithresh_s1 = 1.1#1.1 + (std - 6.)*(2.9 - 2.1)/(48. - 6.)
    ithresh_s2 = 1.7 + (std - 6.)*(0.7 - 1.7)/(48. - 6.)

    #
    # -- 1st step --
    #

    # -- run search --
    inds = align_fxn(burst,ps,std)

    # -- get expanded values --
    eburst = expand_burst(burst,inds,ps)

    # -- denoise --
    dref,dburst = run_denoise_step(eburst,eburst,var,var,ithresh_s1,1)
    dref = expanded_to_burst(dref[:,None],h,w,ps)[:,0]
    dburst = expanded_to_burst(dburst,h,w,ps)

    #
    # -- 2nd step --
    #

    # -- est noise --
    est_std = ndimage.standard_deviation(dframes.cpu().numpy())
    varB = est_std**2

    # -- run search --
    inds = align_fxn(dburst,ps,std)

    # -- get expanded values --
    edburst = expand_burst(dburst,inds,ps)

    # -- denoise --
    dref,dburst = run_denoise_step(eburst,edburst,var,varB,ithresh_s2,1)
    dref = expanded_to_burst(dref[:,None],h,w,ps)[:,0]
    dburst = expanded_to_burst(dburst,h,w,ps)

    return dref,dframes


def run_denoise_step(noisy,basic,var,varB,ithresh,step):

    # -- params --
    rank = 49
    sigma2 = var
    sigmab2 = varB
    device = noisy.device

    # -- compute v.s. ave --
    hw,nftrs,nframes = noisy.shape
    noisy_ave = torch.mean(noisy,dim=-1)[...,None]
    basic_ave = torch.mean(basic,dim=-1)[...,None]
    # dnoisy = noisy - noisy_ave
    dbasic = basic - basic_ave

    # -- covs --
    # covs_noisy = torch.matmul(dnoisy,dnoisy.transpose(1,2))/nframes
    covs_basic = torch.matmul(dbasic,dbasic.transpose(1,2))/nframes
    covs_basic = covs_basic.cpu()#to("cuda:0")

    # -- eign -- 
    # nEigs,nVecs = torch.linalg.eigh(covs_noisy)
    bEigs,bVecs = torch.linalg.eigh(covs_basic)
    bEigs,bVecs = bEigs.to(device),bVecs.to(device)

    # -- modifiy eigs for filter --
    bEigs = mod_eigs(bEigs,nftrs,nframes,sigma2,sigmab2,ithresh)[:,:rank]

    # -- sum over eigenvalues --
    bRankVar = torch.sum(bEigs,dim=1)
    
    # -- create output ! --
    dframes,dref = eigen_denoiser(eigVecs,eigVals)

    return dref,dframes


