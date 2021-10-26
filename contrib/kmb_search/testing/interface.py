
# -- python imports --
import time
import torch
import faiss
import contextlib
import numpy as np
from PIL import Image
from einops import rearrange,repeat
from easydict import EasyDict as edict
import scipy.stats as stats

# -- plotting imports --
import matplotlib
matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt

# -- project imports --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from pyutils import save_image
from align import nnf 
from align.xforms import pix_to_blocks,align_from_flow
from datasets.transforms import get_dynamic_transform

# -- faiss-python imports --
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
from torch_utils import swig_ptr_from_FloatTensor,using_stream
import nnf_utils as nnf_utils
import bnnf_utils as bnnf_utils
import sub_burst as sbnnf_utils
from bp_search import runBpSearch
from nnf_share import padAndTileBatch,padBurst,tileBurst,pix2locs,get_swig_ptr
from kmb_search import runKmSearch

# -- local imports --
from kmb_search.testing.utils import compute_gt_burst,set_seed

def init_zero_tensors(k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,nsiters,device):
    """

    creates all of the zero tensors required for testing

    """

    # -- create zero tensors --
    dtype = torch.float
    burst = torch.zeros(c,t,h,w).type(dtype).to(device)
    init_blocks = torch.zeros(2,t,h,w).type(torch.int).to(device)
    search_frames = torch.zeros(nsiters,nfsearch).type(torch.int).to(device)
    search_ranges = torch.zeros(2,t,nbsearch,h,w).type(torch.int).to(device)
    outDists = torch.zeros(k,h,w).type(torch.float).to(device)
    outInds = torch.zeros(2,t,k,h,w).type(torch.int).to(device)
    modes = torch.zeros(10).type(torch.float).to(device)
    blocks = torch.zeros(2,t,nblocks,h,w).type(torch.int).to(device)
    km_dists = torch.zeros(t,t,nblocks,h,w).type(dtype).to(device)
    centroids = torch.zeros(c,kmeansK,nblocks,h,w).type(dtype).to(device)
    clusters = torch.zeros(t,nblocks,h,w).type(torch.int).to(device)
    cluster_sizes = torch.zeros(kmeansK).type(torch.int).to(device)
    ave = torch.zeros(c,nblocks,h,w).type(torch.float).to(device)

    # -- dict output --
    rdict = edict()
    rdict.dtype = dtype
    rdict.burst = burst
    rdict.init_blocks = init_blocks
    rdict.search_frames = search_frames
    rdict.search_ranges = search_ranges
    rdict.outDists = outDists
    rdict.outInds = outInds
    rdict.modes = modes
    rdict.blocks = blocks
    rdict.km_dists = km_dists
    rdict.centroids = centroids
    rdict.clusters = clusters
    rdict.cluster_sizes = cluster_sizes
    rdict.ave = ave

    # -- include shapes --
    rdict.shapes = edict()
    for key in rdict.keys():
        if key in ["shapes","dtype"]: continue
        rdict.shapes[key] = rdict[key].shape

    return rdict

def exec_test(test_type,test_case,k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,
              kmeansK,std,burst,init_blocks,search_frames,search_ranges,
              outDists,outInds,modes,km_dists,centroids,clusters,
              cluster_sizes,blocks,ave):
              
    # -- contiguous tensors --
    burst = burst.contiguous()
    init_blocks = init_blocks.contiguous()
    search_frames = search_frames.contiguous()
    search_ranges = search_ranges.contiguous()
    outDists = outDists.contiguous()
    outInds = outInds.contiguous()
    modes = modes.contiguous()
    km_dists = km_dists.contiguous()
    centroids = centroids.contiguous()
    clusters = clusters.contiguous()
    blocks = blocks.contiguous()
    ave = ave.contiguous()
    cluster_sizes = cluster_sizes.contiguous()

    # -- extract swig ptrs --
    burst_ptr,dtype = get_swig_ptr(burst,rtype=True)
    init_blocks_ptr = get_swig_ptr(init_blocks)
    search_frames_ptr = get_swig_ptr(search_frames)
    search_ranges_ptr = get_swig_ptr(search_ranges)
    outDists_ptr = get_swig_ptr(outDists)
    outInds_ptr = get_swig_ptr(outInds)
    modes_ptr = get_swig_ptr(modes)
    km_dists_ptr = get_swig_ptr(km_dists)
    centroids_ptr = get_swig_ptr(centroids)
    clusters_ptr = get_swig_ptr(clusters)
    blocks_ptr = get_swig_ptr(blocks)
    ave_ptr = get_swig_ptr(ave)
    cluster_sizes_ptr = get_swig_ptr(cluster_sizes)

    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    print("int sizes.")
    print(k,t,h,w,nblocks)

    # -- setup args --
    args = faiss.GpuTestParams()
    args.test_type = test_type
    args.test_case = test_case
    args.k = k
    args.t = t
    args.h = h
    args.w = w
    args.c = c
    args.ps = ps
    args.nblocks = nblocks
    args.nbsearch = nbsearch # num blocks search
    args.nfsearch = nfsearch # num frames search
    args.kmeansK = kmeansK
    args.std = std
    args.dtype = dtype
    args.burst = burst_ptr
    args.init_blocks = init_blocks_ptr
    args.search_ranges = search_ranges_ptr
    args.outDistances = outDists_ptr
    args.outIndices = outInds_ptr
    args.modes = modes_ptr
    args.km_dists = km_dists_ptr
    args.centroids = centroids_ptr
    args.clusters = clusters_ptr
    args.cluster_sizes = cluster_sizes_ptr
    args.blocks = blocks_ptr
    args.ave = ave_ptr

    # -- choose to block with or without stream --
    with using_stream(res):
        faiss.runKmBurstTest(res, args)

    
