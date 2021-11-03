"""
Some non-local means denoising method.
These methods will improve if the image registration quality is higher.

"""


# -- import search --
from kmb_search import runKmSearch

def run_est_std(warped,weights):
    pass

def compute_denoised_frames(warped,weights):
    pass

def runKmbNlm(burst,ps,nsearch_xy=3,gt_std=None):
    
    # -- init with shapes and such --
    t,i,c,h,w = burst.shape
    ref = t//2

    # -- exec alignment --
    est_std = run_est_std(burst) if gt_std is None else gt_std
    vals,locs = runKmSearch(burst,ps,nsearch_xy,k=1,std=est_std,
                            l2_patchsize=ps,l2_k=5,search_space=None,
                            ref=ref,python=True,gt_info=None)
    
    # -- denoise burst --
    warped = warp_from_locs(burst,locs)
    weights = weights_from_vals(vals)
    dref,dframes = compute_denoised_frames(warped,weights)

    return dref,dframes
    
