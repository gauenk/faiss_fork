
"""
Python calling the C++ API
for KMeans+Burst Search

"""

# -- python --
from einops import rearrange,repeat

# -- pytorch --
import torch
import torchvision.transforms.functional as tvF

# -- faiss --
import faiss
from nnf_share import padBurst,get_optional_device,getImage,getVals,getLocs,get_swig_ptr
from torch_utils import using_stream
from .utils import jitter_search_ranges
from .python_impl import run_kmb_python

def runKmBurstSearch(burst, patchsize, nsearch, k=1, kmeansK = 3,
                     std = None, ref = None, search_ranges=None,
                     img_shape=None, to_flow=False,
                     nsiters=None,nfsearch=None,
                     fmt=False, python=False, gt_info=None):
    
    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    if img_shape is None: img_shape = [c,h,w]
    device = burst.device
    if ref is None: ref = nframes//2
    if std is None: std = 0.

    # -- compute search across image burst --
    vals,locs,modes = [],[],[]
    for i in range(nimages):

        # -- get batch's burst --
        burst_i = burst[:,i]

        # -- get search ranges for image if exists --
        search_ranges_i = search_ranges
        if not(search_ranges is None): search_ranges_i = search_ranges[i]

        # -- execute over search space! --
        if python == True:
            vals_i,locs_i,modes_i = _runKmBurstSearchPython(res, burst_i,
                                                            patchsize, nsearch,
                                                            k = k, kmeansK = kmeansK,
                                                            std = std, ref = ref,
                                                            nsiters=nsiters,
                                                            nfsearch=nfsearch,
                                                            search_ranges=search_ranges_i,
                                                            gt_info=gt_info)
        else:
            vals_i,locs_i,modes_i = _runKmBurstSearch(res, burst_i, patchsize, nsearch,
                                                      k = k, kmeansK = kmeansK,
                                                      std = std, ref = ref,
                                                      search_ranges=search_ranges_i)
        vals.append(vals_i)
        locs.append(locs_i)
        modes.append(modes_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=0)
    # (nframes, nimages, h, w, k, two)
    modes = torch.stack(modes,dim=0)

    # if to_flow:
    #     locs_y = locs[...,0]
    #     locs_x = locs[...,1]
    #     locs = torch.stack([locs_x,-locs_y],dim=-1)
    
    # if fmt:
    #     vals = rearrange(vals,'i h w k -> i (h w) k').cpu()
    #     locs = rearrange(locs,'t i h w k two -> k i (h w) t two').cpu().long()

    return vals,locs,modes

def _runKmBurstSearchPython(res, burst , patchsize, nsearch,
                            k = 1, kmeansK = 3, std = None, ref = None,
                            nsiters=None,nfsearch=None,search_ranges=None,
                            gt_info=None):
    vals,locs,modes = run_kmb_python(res, burst, patchsize, nsearch, k,
                                     kmeansK, std, ref, search_ranges,
                                     nsiters=nsiters, nfsearch=nfsearch,
                                     gt_info=gt_info)
    # vals = vals.cpu()
    # locs = rearrange(locs,'two t k h w -> k t h w two').cpu()
    return vals,locs,modes

def _runKmBurstSearch(res, burst , patchsize, nsearch,
                      k = 1, kmeansK = 3, std = None, ref = None,
                      search_ranges=None):
    
    

    # ----------------------
    #
    #    init none vars
    #
    # ----------------------

    nframes,c,h,w = burst.shape
    if ref is None: ref = nframes//2
    if std is None: raise ValueError("Uknown std -- must be a float.")
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    psHalf = patchsize//2

    # ----------------------
    #
    #     prepare data
    #
    # ----------------------

    # -- burst --
    # burstPad = tvF.pad(burst,(psHalf,)*4,padding_mode="reflect")
    # burstPad = rearrange(burstPad,'t c h w -> c t h w')
    # burstPad = burstPad.contiguous()
    burst_ptr,burst_type = get_swig_ptr(burst,rtype=True)

    # -- init blocks --
    init_blocks = torch.zeros((nframes,h,w)).type(torch.int).to(device).contiguous()
    init_blocks_ptr = get_swig_ptr(init_blocks)

    # -- search --
    if search_ranges is None:
        print("Creating jitter search ranges.")
        search_ranges = jitter_search_ranges(3,nframes,h,w)
    sranges_ptr = get_swig_ptr(search_ranges.to(device))
    print("search_ranges.")
    print(search_ranges.shape)

    # -- vals --
    vals = torch.zeros((k,h,w)).to(device)
    vals_ptr = get_swig_ptr(vals)

    # -- locs --
    locs = torch.zeros((2,nframes,k,h,w)).to(device).type(torch.int)
    locs_ptr,locs_type = get_swig_ptr(locs,rtype=True)
    # print("burstPad.shape: ",burstPad.shape)
    

    # ----------------------
    #
    #   setup C++ interface
    #
    # ----------------------
    print("nsearch: ",nsearch)

    args = faiss.GpuKmBurstParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.t = nframes
    args.ps = patchsize
    args.nsearch = nsearch
    args.kmeansK = kmeansK
    args.std = std # noise level
    args.burst = burst_ptr
    args.dtype = burst_type
    args.init_blocks = init_blocks_ptr
    args.search_ranges = sranges_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # ----------------------
    #
    #   Choosing Stream
    #
    # ----------------------
    
    if is_tensor:
        with using_stream(res):
            faiss.bfKmBurst(res, args)
    else:
        faiss.bfKmBurst(res, args)

    # ---------------------
    #
    #   Reshape for Output
    #
    # ---------------------

    locs = rearrange(locs,'two t k h w -> k t h w two')
    modes = torch.Tensor([])

    return vals, locs, modes

