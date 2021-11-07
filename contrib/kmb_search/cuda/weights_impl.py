
"""
Compute a weight associated with 
each pixel of each burst. This 
allows for "clustering" without the 
actual copying in memory 
which would be silly.

"""


# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda

# -- clgen --
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils import get_img_coords

# -- faiss --
# sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from bp_search import create_mesh_from_ranges
from warp_utils import warp_burst_from_locs,warp_burst_from_pix
th_pad = torchvision.transforms.functional.pad

def compute_weights(sframes,nframes):

    # -- shapes --
    device = sframes.device

    # -- init weights --
    weights = torch.zeros(nframes).to(device)

    # -- fill with sframes --
    weights[sframes] = 1.

    return weights

