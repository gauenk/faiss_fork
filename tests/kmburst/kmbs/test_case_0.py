
# -- python --
import pytest,time,sys
from easydict import EasyDict as edict
from einops import rearrange,repeat
from pyutils import save_image
from pyutils.images import images_to_psnrs,images_to_psnrs_crop

# -- pytorch --
import torch

# -- faiss --
import faiss
import nnf_utils as nnf_utils
from warp_utils import warp_burst_from_locs,pix2locs
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
from kmb_search import runKmSearch,compute_mode_pairs
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.kmbsearch_utils import kmbsearch_setup

VERBOSE=False
def vprint(*args,**kwargs):
    if VERBOSE: print(*args,**kwargs)

@pytest.mark.kmbs
@pytest.mark.kmbs_case0
def test_case_0():

    # -- params --
    k = 1
    t = 4
    h = 16
    w = 16
    c = 3
    ps = 3
    nframes = t
    nsiters = 2 # num search iters
    kmeansK = 3
    nsearch_xy = 3
    nfsearch = 3 # num of frames searched (per iter)
    nbsearch = nsearch_xy**2 # num blocks searched (per frame)
    nblocks = nbsearch**(kmeansK-1)
    std = 20.
    device = 'cuda:0'
    seed = 234
    verbose = False
    tol = 1e-6
    inputs = [k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,nsiters,device]
    zinits = init_zero_tensors(*inputs)
    vprint(zinits.shapes)

    # -- create tensors --
    setup_output = kmbsearch_setup(k,t,h,w,c,ps,std,device)
    burst,clean,inds_gt,flows_gt,blocks_gt = setup_output
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nframes,nfsearch,nsiters,t//2).to(device)
    search_inds = mesh_from_ranges(search_ranges,search_frames[0],inds_gt,t//2).to(device)

    # -- execute test --
    exec_search_test(k,t,h,w,c,ps,nblocks,nsearch_xy,nbsearch,nfsearch,kmeansK,
                     std,burst,clean,inds_gt,flows_gt,device)
    
    # -- compare results --
    # delta = torch.sum(torch.abs(inds_gt - inds)).item()
    # assert delta < 1e-8, "Difference must be smaller than tolerance."

def exec_search_test(k,t,h,w,c,ps,nblocks,nsearch_xy,
                     nbsearch,nfsearch,kmeansK,std,burst,
                     clean,inds_gt,flows_gt,device):

    # -- outputs --
    vals = edict()
    inds = edict()
    times = edict()
    
    # -- add ground-truth --
    vals.gt = torch.zeros(t,h,w).to(device)
    inds.gt = inds_gt
    times.gt = 0

    # -- exec l2 paired search --
    output = nnf_search_and_warp(burst,clean,ps,nsearch_xy,std,vals,inds,times)
    vals.l2,inds.l2,times.l2 = output

    # -- exec kmeans-burst search --
    output = kmb_search_and_warp(clean,clean,ps,nsearch_xy,std,vals,inds,times)
    vals.kmb_clean,inds.kmb_clean,times.kmb_clean = output


    # -- exec kmeans-burst search --
    output = kmb_search_and_warp(burst,clean,ps,nsearch_xy,std,vals,inds,times)
    vals.kmb,inds.kmb,times.kmb = output

    # -- print report --
    print_report(vals,inds,times,burst,clean,ps)

def nnf_search_and_warp(burst,clean,patchsize,nsearch_xy,std,
                        agg_vals,agg_inds,agg_times):

    # -- extract into --
    device = burst.device
    c,t,h,w = burst.shape

    # -- exec search --
    valMean = 0.#compute_mode_pairs(std/255.,c,patchsize)
    burst_tc = rearrange(burst,'c t h w -> t 1 c h w')
    start_time = time.perf_counter()    
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(burst_tc, patchsize, nsearch_xy,
                                              1, valMean = valMean)
    runtime = time.perf_counter() - start_time

    # -- post process data --
    vals = nnf_vals[:,0]
    inds = pix2locs(nnf_locs)
    inds = rearrange(inds,'t 1 h w 1 two -> two t h w')

    # -- to device --
    vals = vals.to(device)
    inds = inds.to(device)    

    return vals,inds,runtime


def kmb_search_and_warp(burst,clean,patchsize,nsearch_xy,std,
                        agg_vals,agg_inds,agg_times):
    # -- extract into --
    device = burst.device
    c,t,h,w = burst.shape

    # -- exec search --
    gt_dist = 0.
    testing = {"ave_choice":1}
    gt_info = {'indices':agg_inds.gt,'clean':clean}
    burst_tc = rearrange(burst,'c t h w -> t 1 c h w')
    start_time = time.perf_counter()    
    _vals,_locs = runKmSearch(burst_tc, patchsize, nsearch_xy, k = 1,
                              std = std/255.,search_space=None,
                              ref=None,mode="python",gt_info=gt_info,
                              testing=testing)
    runtime = time.perf_counter() - start_time

    # -- post process data --
    inds = rearrange(_locs,'1 1 t h w two -> two t h w')
    vals = _vals[0]
    print("inds.shape: ",inds.shape)

    # -- to device --
    vals = vals.to(device)
    inds = inds.to(device)

    return vals,inds,runtime

def print_report(vals,inds,times,burst,clean,patchsize):

    # -- report bools --
    vreport = edict()
    vreport.runtimes = True
    vreport.delta_inds = False
    vreport.psnrs = True

    # -- methods --
    methods = list(inds.keys())
    
    # -- time --
    if vreport.runtimes:
        print("\n"*3)
        print("-=-=-=- Runtimes -=-=-=-")
        for method in methods:
            runtime = times[method]
            print(f"[%s]: %2.3e" % (method,runtime))
    
    # -- compare with flow --
    if vreport.delta_inds:
        print("\n"*3)
        print("-=-=-=- Delta Flow -=-=-=-")
        nframes = vals.gt.shape[0]
        for method in methods:
            if method == "gt": continue
            print(f"-=-=-=- Method [{method}] -=-=-=-")
            for t in range(nframes):
                print(inds[method].shape,inds.gt.shape)
                delta = torch.abs(inds[method][:,t] - inds.gt[:,t])
                delta = delta.type(torch.float).mean()
                print("[%d] Delta: %2.3f" %(t,delta))
    
    # -- psnrs --
    if vreport.psnrs:
        print("\n"*3)
        print("-=-=-=- PSNRS -=-=-=-")
        t,ps = clean.shape[1],patchsize
        repimg = repeat(clean[:,t//2],'c h w -> t c h w',t=t)
        for method in methods:
            m_inds = inds[method]
            warp = warp_burst_from_locs(clean,m_inds[:,:,None])
            warp = rearrange(warp,'1 c t h w -> t c h w')
            psnrs = images_to_psnrs_crop(warp,repimg,ps)
            print(f"-- [{method}] --")
            print(psnrs)
            save_image(f"tkmb_{method}.png",warp)
    
