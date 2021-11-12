
# -- python --
import pytest,time,sys
from easydict import EasyDict as edict
from einops import rearrange,repeat
from pyutils import save_image
from pyutils.images import images_to_psnrs,images_to_psnrs_crop
from pathlib import Path
from PIL import Image

# -- pytorch --
import torch
import numpy as np

# -- faiss --
import faiss
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
import nnf_utils as nnf_utils
from nnf_utils import runNnfBurst
from warp_utils import *
# from warp_utils import warp_burst_from_locs,pix2locs,warp_burst_from_pix
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
from kmb_search import runKmSearch,compute_mode_pairs
from kmb_nlm import run_nlm#runKmbNlm,runL2Nlm
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.kmbsearch_utils import kmbsearch_setup

VERBOSE=False
def vprint(*args,**kwargs):
    if VERBOSE: print(*args,**kwargs)

def divUp(a,b): return (a-1)//b+1

def read_burst(path,nframes):
    burst = []
    for t in range(nframes-1):
        full_path = str(path / ("deno_%03d.png" % (t+1)))
        img = np.array(Image.open(full_path).convert("RGB"))
        burst.append(img)
    burst = np.stack(burst)
    burst = torch.FloatTensor(burst)/255.
    burst = rearrange(burst,'t h w c -> t c h w')

    mb = burst.max().item()
    burst = (burst - burst.min())/mb

    return burst

@pytest.mark.kmb_nlm
@pytest.mark.kmb_nlm_case0
def test_case_0():

    # -- params --
    k = 1
    t = 10
    h = 32
    w = 32
    c = 3
    ps = 3
    nframes = t
    nsiters = 2 # num search iters
    kmeansK = 3
    nsearch_xy = 3
    nfsearch = 3 # num of frames searched (per iter)
    nbsearch = nsearch_xy**2 # num blocks searched (per frame)
    nblocks = nbsearch**(kmeansK-1)
    std = 50.
    device = 'cuda:0'
    seed = 234
    verbose = False
    tol = 1e-6
    inputs = [k,t,h,w,c,ps,nblocks,nbsearch,nfsearch,kmeansK,nsiters,device]
    zinits = init_zero_tensors(*inputs)
    vprint(zinits.shapes)

    # -- create tensors --
    setup_output = kmbsearch_setup(k,t,h,w,c,ps,std,device)
    noisy,clean,inds_gt,flows_gt,blocks_gt = setup_output
    search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    search_frames = tiled_search_frames(nframes,nfsearch,nsiters,t//2).to(device)
    search_inds = mesh_from_ranges(search_ranges,search_frames[0],inds_gt,t//2).to(device)
    save_image("burst.png",noisy.transpose(0,1))

    # -- compare with vnlmb -- 
    path = Path("/home/gauenk/Documents/faiss/tests/vnlb_samples/")
    vburst = read_burst(path,nframes)
    # refburst = repeat(clean[:,t//2],'c h w -> c t h w')
    clean_tc = clean.transpose(0,1)
    print(vburst.shape,clean_tc.shape)
    psnrs = images_to_psnrs_crop(vburst,clean_tc[1:],2*divUp(ps,2))
    print(" -- VNLMB Denoising PSNRS --")
    print(psnrs)
    print(np.mean(psnrs))

    # noisy_t = noisy.transpose(0,1)
    # for t in range(nframes):
    #     save_image("frame-%03d.png"%t,noisy_t[[t]])
        
    # -- execute test --
    exec_search_test(k,t,h,w,c,ps,nblocks,nsearch_xy,nbsearch,nfsearch,kmeansK,
                     std,noisy,clean,inds_gt,flows_gt,device)
    
    # -- compare results --
    # delta = torch.sum(torch.abs(inds_gt - inds)).item()
    # assert delta < 1e-8, "Difference must be smaller than tolerance."

def exec_search_test(k,t,h,w,c,ps,nblocks,nsearch_xy,
                     nbsearch,nfsearch,kmeansK,std,noisy,
                     clean,inds_gt,flows_gt,device):

    # -- outputs --
    dimgs = edict()
    inds = edict()
    times = edict()
    desc = edict()
    
    # ----------------------------
    #
    #       Perfect Methods
    #
    # ----------------------------

    # -- add ground-truth --
    dimgs.gt = clean[:,t//2]
    times.gt = 0
    inds.gt = run_nnf_search(clean,clean,ps,nsearch_xy,std)

    # ----------------------------
    #
    #    Actual Search Methods
    #
    # ----------------------------

    # -- exec l2 paired search --
    params = {"nnf_itype":"noisy"}
    output = run_nnf_nlm(noisy,clean,ps,nsearch_xy,std,inds.gt,params)
    dimgs.l2,inds.l2,times.l2 = output
    desc.l2 = "standard searching"

    # -- exec l2 paired search --
    params = {"nnf_itype":"clean"}
    output = run_nnf_nlm(noisy,clean,ps,nsearch_xy,std,inds.gt,params)
    dimgs.l2_clean,inds.l2_clean,times.l2_clean = output
    desc.l2_clean = "perfect searching"

    # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"given","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2","sranges_type":"l2"}
    # output = run_kmb_nlm(noisy,clean,ps,nsearch_xy,std,inds.gt,testing)
    # dimgs.kmb,inds.kmb,times.kmb = output
    # desc.kmb = "actual searching."

    # -- print report --
    print_report(dimgs,inds,times,noisy,clean,ps)

def run_nnf_nlm(noisy,clean,patchsize,nsearch_xy,std,inds_gt,params):

    # -- extract into --
    device = noisy.device
    c,t,h,w = noisy.shape
    assert c == 3,"color channel is 3."

    # -- gt info --
    gt_info = {"clean":clean}

    # -- exec search --
    # noisy_tc = rearrange(noisy,'c t h w -> t c h w')
    start_time = time.perf_counter()    
    denoised,inds = run_nlm(noisy,patchsize,std/255.,
                            align_method="l2",
                            gt_info=gt_info,
                            params=params)
    # denoised,inds = runL2Nlm(noisy, patchsize, 11, nsearch_xy, gt_std=std/255.,
    #                          gt_info=gt_info, params=params)
    runtime = time.perf_counter() - start_time

    # -- to device --
    denoised = denoised.to(device)

    return denoised,inds,runtime


def run_kmb_nlm(noisy,clean,patchsize,nsearch_xy,std,inds_gt,params):

    # -- extract into --
    device = noisy.device
    c,t,h,w = noisy.shape
    assert c == 3,"color channel is 3."

    # -- exec search --
    gt_dist = 0.
    gt_info = {'indices':inds_gt,'clean':clean}
    # noisy_tc = rearrange(noisy,'c t h w -> t c h w')
    start_time = time.perf_counter()    
    denoised,inds = run_nlm(noisy,patchsize,std/255.,
                            align_method="kmb",
                            gt_info=gt_info,
                            params=params)
    runtime = time.perf_counter() - start_time

    # -- to device --
    denoised = denoised.to(device)

    return denoised,inds,runtime

def print_report(dimgs,inds,times,noisy,clean,patchsize):

    # -- report bools --
    vreport = edict()
    vreport.runtimes = True
    vreport.align_psnrs = True
    vreport.dimg_psnrs = True

    # -- methods --
    methods = list(dimgs.keys())
    
    # -- time --
    if vreport.runtimes:
        print("\n"*3)
        print("-=-=-=- Runtimes -=-=-=-")
        for method in methods:
            runtime = times[method]
            print(f"[%s]: %2.3e" % (method,runtime))
    
    # -- psnrs --
    if vreport.align_psnrs:
        print("\n"*3)
        print("-=-=-=- Alignment PSNRS -=-=-=-")
        t,ps = clean.shape[1],patchsize
        repimg = repeat(clean[:,t//2],'c h w -> t c h w',t=t)
        for method in methods:
            print_method = method_rename(method)
            m_inds = inds[method]
            warp = warp_burst_from_pix(clean,m_inds[:,:,None])
            warp = rearrange(warp,'1 c t h w -> t c h w')
            psnrs = images_to_psnrs_crop(warp,repimg,2*divUp(ps,2))
            print(f"-- [{print_method}] --")
            print(psnrs)
            save_image(f"tkmb_align_{method}.png",warp)
    
    # -- denoiser psnrs --
    if vreport.dimg_psnrs:
        print("\n"*3)
        print("-=-=-=- Denoising PSNRS -=-=-=-")
        t,ps = clean.shape[1],patchsize
        ref_img = clean[:,t//2]
        for method in methods:
            print_method = method_rename(method)
            denoised = dimgs[method]
            psnrs = images_to_psnrs_crop(denoised,ref_img,2*divUp(ps,2))
            print(f"-- [{print_method}] --")
            print(psnrs)
            save_image(f"tkmb_dimg_{method}.png",denoised)
    

def run_nnf_search(noisy,clean,patchsize,nsearch_xy,std):

    # -- extract into --
    device = noisy.device
    c,t,h,w = noisy.shape

    # -- exec search --
    noisy_tc = rearrange(noisy,'c t h w -> t 1 c h w')
    start_time = time.perf_counter()    
    valMean = 2*std/255.
    nnf_vals,nnf_locs = runNnfBurst(noisy_tc, patchsize, nsearch_xy,
                                    1, valMean = valMean)
    runtime = time.perf_counter() - start_time

    # -- post process data --
    vals = nnf_vals[:,0]
    # inds = flow2pix(nnf_locs)
    inds = torch.flip(nnf_locs,dims=(-1,))
    inds = rearrange(inds,'t 1 h w 1 two -> two t h w')

    # -- to device --
    vals = vals.to(device)
    inds = inds.to(device)    

    return inds

def method_rename(method):
    if method == "kmb_v9":
        return "Ref Centroid"
    elif method == "kmb_v13":
        return "Ref Clean"
    elif method == "kmb_v14":
        return "Ref Clean + Noise"
    elif method == "kmb_v15":
        return "Ref Noisy"
    else:
        return method

        
