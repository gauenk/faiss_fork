

import torch
from .utils import get_optional_field,parse_ctype

def get_score_function(testing):
    choice = get_optional_field(testing,"score","weighted_ave_v1")
    if choice in ["ave","ave_v1"]:
        return get_ave_v1()
    if choice == "ave_v2":
        return get_ave_v2()
    elif choice == "weighted_ave_v1":
        return get_weighted_ave_v1()
    elif choice == "weighted_ave_v2":
        return get_weighted_ave_v2()
    else:
        raise ValueError(f"Uknown ave function [{choice}]")

def get_ave_v1():
    def ave_impl(l2_vals,modes,sizes,nframes):
        mvals = torch.nansum(torch.abs(l2_vals-modes),dim=0)/nframes
        return mvals
    return ave_impl
        
def get_ave_v2():
    def ave_impl(l2_vals,modes,sizes,nframes):
        mvals = torch.nanmean(torch.abs(l2_vals-modes),dim=0)
        return mvals
    return ave_impl

def get_weighted_ave_v1():
    def weighted_ave_v1(l2_vals,modes,sizes,nframes):
        scores = torch.nansum(torch.abs(l2_vals-modes)*sizes,dim=0)
        scores = scores/nframes
        return scores
    return weighted_ave_v1

def get_weighted_ave_v2():
    def weighted_ave_v2(l2_vals,modes,sizes,nframes):
        mvals = torch.nanmean(torch.abs(l2_vals-modes)*sizes,dim=0)
        return mvals
    return weighted_ave_v2
