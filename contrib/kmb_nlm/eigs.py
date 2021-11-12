import torch

def paul_var(eigs,nftrs,nframes,sigma2,sigmab2):

    # -- gamma --
    # gamma = nftrs/float(nframes)
    gamma = 0.95
    thresh = sigmab2 * (1 + np.sqrt(gamma))**2

    # -- thresh --
    inds = torch.where(eigs > thresh)

    # -- inv --
    tmp = eigs[inds] - sigmab2*(1.+gamma)
    tmp2 = tmp*tmp
    inv = 1 - 4.*gamma*sigmab2*sigmab2/tmp2
    rinds = torch.where(inv <= 0)
    inv[rinds] = 0
    inv = torch.sqrt(inv)

    # -- final --
    eigs[inds] = tmp * 0.5 * inv
    
    # eigs[inds] = tmp * 0.5 * (1. + np.sqrt(max(0.,1 - 4.*gamma*sigmab2*sigmab2/tmp2)))

    # -- zeros -- 
    inds = torch.where(eigs <= thresh)
    eigs[inds] = 0

def mod_eigs(eigs,nftrs,nframes,sigma2,sigmab2,ithresh):


    #
    # -- filter eigs --
    #

    paul_var(eigs,nftrs,nframes,sigma2,sigmab2)


    #
    # -- bayes filter --
    #

    # mat.covEigVals[k] = (mat.covEigVals[k] > thres * sigma2) ?
    # 1.f / ( 1. + sigma2 / mat.covEigVals[k] ) : 0.f;
    inv_thresh = ithresh * sigma2
    inds = torch.where(eigs > inv_thresh)
    eigs[inds] = 1. / ( 1. +  sigma2 / eigs[inds] )
    inds = torch.where(eigs < inv_thresh)
    eigs[inds] = 0.

    # print("All Zero? ",torch.all(eigs<1e-8).item())
    # print(sigma2,ithresh*sigma2,thresh)

    return eigs
