
import torch
from .utils import get_optional_field,parse_ctype
from .compute_mode_impl import compute_mode

def get_mode_function(testing):
    choice = get_optional_field(testing,"mode","zeros")
    if choice == "zeros":
        return get_zero_modes()
    elif choice == "centroids":
        return get_centroid_modes()
    else:
        raise ValueError("Uknown mode function [{choice}]")

def get_zero_modes():
    def zero_modes(std,c,ps,sizes):
        modes = compute_mode(std,c,ps,sizes,type='centroids')
        modes = torch.zeros_like(modes)
        return modes
    return zero_modes

def get_centroid_modes():
    def centroid_modes(std,c,ps,sizes):
        modes = compute_mode(std,c,ps,sizes,type='centroids')
        return modes
    return centroid_modes
 

