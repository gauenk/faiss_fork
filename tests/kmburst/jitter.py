"""
Test KmBurst using only local jitter

"""

# -- python --
import time,sys,pytest
import torch
import faiss
import contextlib
import numpy as np
from PIL import Image
from einops import rearrange,repeat
from easydict import EasyDict as edict
import scipy.stats as stats

# -- project --

# -- faiss --
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from torch_utils import swig_ptr_from_FloatTensor,using_stream
import nnf_utils as nnf_utils
import bnnf_utils as bnnf_utils
import sub_burst as sbnnf_utils
from bp_search import runBpSearch
from nnf_share import padAndTileBatch,padBurst,tileBurst,pix2locs,warp_burst_from_locs
from kmb_search import runKmSearch
from kmb_search.testing.utils import compute_gt_burst,set_seed

@pytest.mark.local_jitter
def test_local_jitter():
    pass
