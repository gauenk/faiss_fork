

import torch
from .utils import get_optional_field,parse_ctype

def get_score_function(testing):
    choice = get_optional_field(testing,"score","ave")
    if choice == "ave":
        return get_ave()
    elif choice == "weighted_ave_v1":
        return get_weighted_ave_v1()
    elif choice == "weighted_ave_v2":
        return get_weighted_ave_v2()
    else:
        raise ValueError("Uknown ave function [{choice}]")

def get_ave():
    def ave_impl(l2_vals,modes,sizes,nframes):
        mvals = torch.nanmean(torch.abs(l2_vals-modes))
        return mvals
    return ave_impl
        
def get_weighted_ave_v1():
    def weighted_ave_v1(l2_vals,modes,sizes,nframes):
        scores = torch.nansum(torch.abs(l2_vals-modes)*sizes,dim=0)
        scores = scores/nframes
        return scores
    return ave_impl

def get_weighted_ave_v2():
    def weighted_ave_v2(l2_vals,modes,sizes,nframes):
        mvals = torch.nanmean(torch.abs(l2_vals-modes)*sizes,dim=0)
        return mvals
    return ave_impl
