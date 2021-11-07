


# -- python --
import math
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- project --
from pyutils import save_image

# -- pytorch --
import torch
import torch.nn.functional as nnF

def pad_first_dim(tensor,pad):
    tensor = torch.transpose(tensor,0,-1)
    tensor = nnF.pad(tensor,(0,pad),value=float("nan"))
    tensor = torch.transpose(tensor,0,-1)
    return tensor

# -------------------------------------------
#
#     Update State with Randomness
#
# -------------------------------------------

def update_state(propDists,prevDists,propModes,prevModes,
                 propInds,prevInds,sframes,s_iter):

    """
    kmb_topk_rand:
    Jump out of local optima by picking
    an element of the search space that is not optimal

    propDist/prevDist ratio:
    Jump out of local optima by picking
    a new distance that was not smaller than the previous distance
    """

    # -- take top 1 of proposed --
    # print("prevInds.shape: ",prevInds.shape)
    propDists,propModes,propInds = kmb_topk_rand(propDists,propModes,propInds,s_iter)

    # -- create modified dists for selection --
    mPropDists = torch.nanmean(torch.abs(propDists-propModes),dim=0)
    mPrevDists = torch.nanmean(torch.abs(prevDists-prevModes),dim=0)
    mPrevDists[torch.where(torch.isnan(mPrevDists))] = 1000.
    mPropDists[torch.where(torch.isnan(mPropDists))] = 1000.
    assert mPrevDists.shape[0] == 1,"k == 1"

    # -- compute ratio --
    prior = 0.90 if s_iter > 1 else 1.
    ratioDists = mPropDists/mPrevDists * prior
    pexp = 2*math.log10(s_iter+1)+1#math.exp(2*math.log10(s+1))
    pexp = pexp if s_iter > 1 else 1000.
    ratioDists = torch.pow(ratioDists,pexp)
    coin_flip = torch.rand_like(ratioDists)
    toChange = torch.where(coin_flip > ratioDists)
    
    # -- init next state --
    nextDists = prevDists.clone()
    nextInds = prevInds.clone()
    nextModes = prevModes.clone()
    # print("nextInds.shape: ",nextInds.shape)
    # print("propInds.shape: ",propInds.shape)
    # print(propInds[:,:,0,9,6])
    # print("nextInds.shape: ",nextInds.shape)
    # print(coin_flip.shape)

    # print(propInds[:,:,:,9,6])
    # -- fill in state --
    # print("pre: ",nextInds[:,:,0,9,6])
    nframes = propInds.shape[1]
    nfsearch = len(sframes)
    for ti,tj in enumerate(sframes):
        nextDists[tj][toChange] = propDists[ti][toChange]
        nextModes[tj][toChange] = propModes[ti][toChange]
    for t in range(nframes):
        nextInds[0,t][toChange] = propInds[0,t][toChange]
        nextInds[1,t][toChange] = propInds[1,t][toChange]
    # print("nextInds.shape: ",nextInds.shape)
    # print("post: ",nextInds[:,:,0,9,6])

    return nextDists,nextModes,nextInds

def kmb_topk_update(propDists,prevDists,propModes,prevModes,
                    propInds,prevInds,propSFrames,prevSFrames):

    # -- pad across first dimension --
    # print("prevDists.shape: ",prevDists.shape)
    # print("propDists.shape: ",propDists.shape)
    # print("propModes.shape: ",propModes.shape)
    # pad = prevDists.shape[0] - propDists.shape[0]
    # propDists = pad_first_dim(propDists,pad)
    # propModes = pad_first_dim(propModes,pad)
    # print("propDists.shape: ",propDists.shape)    
    # print("propModes.shape: ",propModes.shape)
    
    # -- insert proposed into prev --
    # b = propDists.shape[1]
    # propDists_raw = propDists.clone()
    # propModes_raw = propModes.clone()
    # propDists = prevDists.clone().repeat(1,b,1,1)
    # propModes = prevModes.clone().repeat(1,b,1,1)
    # propDists[propSFrames] = propDists_raw
    # propModes[propSFrames] = propModes_raw

    # -- create stacks --
    aug_vals = torch.cat([prevDists,propDists],dim=1)
    aug_modes = torch.cat([prevModes,propModes],dim=1)
    aug_inds = torch.cat([prevInds,propInds],dim=2)
    
    # -- exec and return --
    K = prevDists.shape[1]
    return kmb_topk(aug_vals,aug_modes,aug_inds,K)

def kmb_topk_rand(vals,modes,inds,s_iter):
    
    # -- init --
    device = vals.device
    tK,s,h,w = vals.shape
    two,t,s,h,w = inds.shape

    # -- run pytorch topk --
    mvals = torch.nanmean(torch.abs(vals - modes),dim=0)

    # -- misc --
    # print(inds[:,:,0,9,6])
    # print(inds[:,:,1,9,6])
    # print(inds[:,:,5,9,6])
    # print(mvals[:,9,6])
    vals_topk,modes_topk,inds_topk = topk_torch_rand(mvals,vals,modes,inds,s_iter)
    # print("inds_topk.shape: ",inds_topk.shape)
    # print(inds_topk[:,:,0,9,6])

    return vals_topk,modes_topk,inds_topk


def kmb_topk(vals,modes,inds,K):

    # -- init --
    device = vals.device
    tK,s,h,w = vals.shape
    two,t,s,h,w = inds.shape

    # -- creat output vars --
    # vals = vals.cpu().numpy()
    # inds = inds.cpu().numpy()
    # vals_topk = np.zeros((K,h,w))
    # inds_topk = np.zeros((two,t,K,h,w))

    # -- run pytorch topk --
    mvals = torch.nanmean(torch.abs(vals - modes),dim=0)
    # print("-- pre top k --")
    # print("inds.shape: ",inds.shape)
    # print("mvals.shape: ",mvals.shape)
    # print(inds[:,:,0,4,5])
    # print(mvals[:,4,5])
    # print(vals[:,:,4,5])
    # print(vals[:,:,4,5].shape)
    vals_topk,modes_topk,inds_topk = topk_torch(mvals,vals,modes,inds,K)
    # print("inds_topk.shape: ",inds_topk.shape)
    # print(inds_topk[:,:,0,9,6])

    # print("-- post top k --")
    # print("vals_topk.shape: ",vals_topk.shape)
    # print(inds_topk[:,:,0,4,5])
    # print(vals_topk[:,:,4,5])

    # -- launch numba --
    # kmb_topk_numba(vals,inds,vals_topk,inds_topk)

    # -- pytorch to numpy --
    # vals_topk = torch.FloatTensor(vals_topk).to(device)
    # inds_topk = torch.IntTensor(inds_topk).to(device)

    return vals_topk,modes_topk,inds_topk

def topk_torch_rand(mvals,vals,modes,inds,s_iter,K=1):
    """
    Jump out of local optima by picking
    an element of the search space that is not optimal
    """

    # -- take min --
    assert K == 1,"only K = 1 right now."
    topk = torch.topk(mvals,K,dim=0,largest=False,sorted=True)
    topk_mvals = topk.values

    # -- take ratio w.r.t. [ideal?] min --
    eps = 1e-8
    ratio_mvals = (topk_mvals+eps) / (mvals+eps) # [0,1] by construction
    
    # -- sample using ratios as weights --
    s,h,w = ratio_mvals.shape
    weights = ratio_mvals
    pexp = 5*math.log10(s_iter+1)+1#math.exp(2*math.log10(s+1))
    pexp = pexp if s_iter > 1 else 1000.
    weights = torch.pow(weights,pexp)

    # -- save weights not equal to 1 on exh. search --
    # wsum = torch.sum(weights,dim=0)
    # wimg = torch.abs(wsum-1.)<1e-5
    # wimg = wimg.type(torch.float)
    # save_image("tkmb_wimg.png",wimg)

    # -- sample across search space using weights --
    weights = rearrange(weights,'s h w -> (h w) s')
    samples = torch.multinomial(weights,1)
    samples = rearrange(samples,'(h w) 1 -> 1 h w',h=h)
    # print("samples.shape: ",samples.shape)
    # print("topk.indices.shape: ",topk.indices.shape)
    # print(ratio_mvals)
    # print(samples)
    # print(topk.indices)

    # -- use indices to 
    return index_topk(vals,modes,inds,K,samples)

def topk_torch(mvals,vals,modes,inds,K):
    # print("pre")
    topk = torch.topk(mvals,K,dim=0,largest=False,sorted=True)
    # torch.cuda.synchronize()
    # print("post")
    return index_topk(vals,modes,inds,K,topk.indices)
    
def index_topk(vals,modes,inds,K,indices):
    two = inds.shape[0]
    assert two == 2,"check [modes,inds] order."
    tK = vals.shape[0]
    vals_topk = torch.zeros_like(vals)[:,:K]
    modes_topk = torch.zeros_like(modes)[:,:K]
    # print(vals.shape,indices.shape,modes.shape,vals_topk.shape)
    # exit()
    # for tk in range(tK):
    #     print(vals_topk.shape,vals.shape)
    #     vals_topk[tk] = torch.gather(vals[tk],dim=0,index=indices)
    #     modes_topk[tk] = torch.gather(modes[tk],dim=0,index=indices)
    # exit()

    inds_topk = torch.zeros_like(inds)[:,:,:K]
    # print("inds.shape: ",inds.shape)
    # print("indices.shape: ",indices.shape)
    # print(inds[:,:,6,4,5])
    # print(indices[0,4,5])
    for i in range(inds.shape[0]):
        for t in range(inds.shape[1]):
            inds_topk[i,t] = torch.gather(inds[i,t],dim=0,index=indices)
    # print(inds_topk[:,:,0,4,5])
    return vals_topk,modes_topk,inds_topk




def kmb_topk_numba(vals,inds,vals_topk,inds_topk):

    pass
    # s,h,w = vals.shape
    # K,h,w = vals_topk.shape

    # for hi in prange(h):
    #     for wi in prange(w):            
    #         for si in range(s):

    #             # -- update --

                
                
