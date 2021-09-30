
import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from nnf_share import *

def getSubAveTorch(subAve,h,w,c,device,t=None):
    if subAve is None:
        subAve = torch.zeros(c, h, w, device=device, dtype=torch.float32)
    else:
        assert subAve.shape == (c, h, w)
    subAve_ptr,subAve_type = torch2swig(subAve)
    return subAve,subAve_ptr,subAve_type

def rows_uniq_elems(a):
    a_sorted = torch.sort(a,axis=-1)
    return a[(a_sorted[...,1:] != a_sorted[...,:-1]).all(-1)]

