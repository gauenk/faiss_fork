
# -- import search --
from kmb_search import runKmSearch as _runKmSearch
from nnf_utils import runNnfBurst as _runNnfBurst
from warp_utils import flow2pix


# -- local --
from .utils import return_optional_params

def get_align_method(method,params):

    if method == "kmb":
        return wrap_kmb_search(params)
    elif method == "l2":
        return wrap_l2_search(params)
    else:
        raise KeyError("Uknown align method [{method}]")
    

def wrap_kmb_search(params):

    nsearch_xy = return_optional_params(params,'nsearch_xy',3)
    k = return_optional_params(params,'k',1)
    gt_info = return_optional_params(params,'gt_info',None)
    sparams = return_optional_params(params,'sparams',None)

    def kmb_search(burst,ps,std):

        # -- init with shapes and such --
        c,t,h,w = burst.shape
        est_std = gt_std
    
        # -- exec alignment --
        # est_std = run_est_std(burst) if gt_std is None else gt_std
        burst_tc = rearrange(burst,'c t h w -> t 1 c h w')
        _vals,_locs = _runKmSearch(burst_tc, ps, nsearch_xy, k = k,
                                   std = est_std,mode="python",
                                   gt_info=gt_info,testing=sparams)
        
        # -- format alignment indices [inds] --
        inds = rearrange(_locs,'1 1 t h w two -> t 1 h w 1 two')
        inds = flow2pix(inds)
        inds = rearrange(inds,'t 1 h w k two -> two t k h w')
        inds = torch.flip(inds,dims=(1,))
    
        return inds

    return kmb_search
        
def wrap_l2_search(params):

    nsearch_xy = return_optional_params(params,'nsearch_xy',3)
    k = return_optional_params(params,'k',1)
    gt_info = return_optional_params(params,'gt_info',None)
    sparams = return_optional_params(params,'sparams',None)

    def l2_search(burst,ps,std):

        # -- init with shapes and such --
        c,t,h,w = burst.shape
    
        # -- exec alignment --
        # est_std = run_est_std(burst) if gt_std is None else gt_std
        est_std = gt_std
        valMean = 2*est_std
        nnf_img = select_search_image(burst,gt_info,sparams)
        nnf_img_tc = rearrange(nnf_img,'c t h w -> t 1 c h w')
        nnf_vals,nnf_locs = _runNnfBurst(nnf_img_tc, search_ps, nsearch_xy, k, valMean)
    
        # -- format alignment indices [inds] --
        vals = nnf_vals[:,0]
        inds = torch.flip(nnf_locs,dims=(-1,))
        inds = rearrange(inds,'t 1 h w k two -> two t k h w')
    
        return inds
    
    return l2_search


