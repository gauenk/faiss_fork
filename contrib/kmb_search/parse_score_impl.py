

import torch
from .utils import get_optional_field
from .fill_utils import fill_ref_l2vals
from .parse_utils import parse_ctype

def get_score_function(testing):
    choice = get_optional_field(testing,"score","weighted_ave_v1")
    ave_ctype = get_optional_field(testing,"ave_centroid_type","noisy")
    filter_fxn = get_filter_values(ave_ctype)
    if choice in ["ave","ave_v1"]:
        return get_ave_v1(filter_fxn)
    if choice == "ave_v2":
        return get_ave_v2(filter_fxn)
    elif choice == "weighted_ave_v1":
        return get_weighted_ave_v1(filter_fxn)
    elif choice == "weighted_ave_v2":
        return get_weighted_ave_v2(filter_fxn)
    else:
        raise ValueError(f"Uknown ave function [{choice}]")

def get_filter_values(ave_ctype):

    def filter_ave_clean(l2_vals,clusters,ref):
        # -- remove if ave is clean --
        if "clean" in ave_ctype:# and "-" not in ave_ctype:
            fill_ref_l2vals(0.,clusters,l2_vals,ref)
            return l2_vals
        else: return l2_vals

    def filter_outliers(l2_vals,sizes):
        thresh = 0.22
        # mask_sizes = sizes[torch.where(l2_vals > thresh)]
        bool_a = l2_vals > thresh
        bool_b = sizes < 2
        bool_ab = torch.logical_and(bool_a,bool_b)
        sizes[torch.where(bool_ab)] = 0
        return l2_vals,sizes

    def filter_values(l2_vals,clusters,sizes,ref):
        l2_vals = filter_ave_clean(l2_vals,clusters,ref)
        # l2_vals,sizes = filter_outliers(l2_vals,sizes)
        return l2_vals,sizes

    return filter_values

def get_ave_v1(filter_fxn):
    def ave_impl(l2_vals,modes,clusters,sizes,nframes,ref):
        l2_vals,sizes = filter_fxn(l2_vals,clusters,sizes,ref)
        mvals = torch.nansum(torch.abs(l2_vals-modes),dim=0)/nframes
        return mvals
    return ave_impl
        
def get_ave_v2(filter_fxn):
    def ave_impl(l2_vals,modes,clusters,sizes,nframes,ref):
        l2_vals,sizes = filter_fxn(l2_vals,clusters,sizes,ref)
        mvals = torch.nanmean(torch.abs(l2_vals-modes),dim=0)
        return mvals
    return ave_impl

def get_weighted_ave_v1(filter_fxn):
    def weighted_ave_v1(l2_vals,modes,clusters,sizes,nframes,ref):
        l2_vals,sizes = filter_fxn(l2_vals,clusters,sizes,ref)
        scores = torch.nansum(torch.abs(l2_vals-modes)*sizes,dim=0)
        # sizes = torch.sum(sizes,dim=0)
        # scores = scores/sizes
        scores = scores/nframes
        return scores
    return weighted_ave_v1

def get_weighted_ave_v2(filter_fxn):
    def weighted_ave_v2(l2_vals,modes,clusters,sizes,nframes,ref):
        l2_vals,sizes = filter_fxn(l2_vals,clusters,sizes,ref)
        mvals = torch.nanmean(torch.abs(l2_vals-modes)*sizes,dim=0)
        return mvals
    return weighted_ave_v2
