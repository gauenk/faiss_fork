
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
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
import nnf_utils as nnf_utils
from warp_utils import *
# from warp_utils import warp_burst_from_locs,pix2locs,warp_burst_from_pix
from kmb_search import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
from kmb_search import runKmSearch,compute_mode_pairs
from kmb_search.testing.interface import exec_test,init_zero_tensors
from kmb_search.testing.kmbsearch_utils import kmbsearch_setup

VERBOSE=False
def vprint(*args,**kwargs):
    if VERBOSE: print(*args,**kwargs)

def divUp(a,b): return (a-1)//b+1


@pytest.mark.kmbs
@pytest.mark.kmbs_case0
def test_case_0():

    # -- params --
    k = 1
    t = 10
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
    std = 100.
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
    vals = edict()
    inds = edict()
    times = edict()
    desc = edict()
    
    # ----------------------------
    #
    #       Perfect Methods
    #
    # ----------------------------

    # -- add ground-truth --
    vals.gt = torch.zeros(t,h,w).to(device)
    inds.gt = inds_gt
    times.gt = 0

    # -- exec l2 paired search --
    output = run_nnf_search(clean,clean,ps,nsearch_xy,std,vals,inds,times)
    vals.l2_clean,inds.l2_clean,times.l2_clean = output

    # -- exec kmeans-burst search --
    # testing = {"nfsearch":2,"ave_centroid_type":"clean","ave":"ref_centroids",
    #            "cluster_centroid_type":"clean","cluster":"fill"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_clean,inds.kmb_clean,times.kmb_clean = output

    # ----------------------------
    #
    #    Actual Search Methods
    #
    # ----------------------------

    # -- exec l2 paired search --
    output = run_nnf_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times)
    vals.l2,inds.l2,times.l2 = output
    desc.l2 = "standard searching"

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"noisy","nfsearch":4,
    #            "cluster":"fill","cluster_centroid_type":"noisy"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb,inds.kmb,times.kmb = output
    # desc.kmb = "searching a burst using the mean w.r.t reference frame"
    # desc.kmb += "maybe same as l2?"

    # -- exec kmeans-burst search --
    # testing = {"ave":"ave_centroids","ave_centroid_type":"clean","nfsearch":4}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v2,inds.kmb_v2,times.kmb_v2 = output

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"clean","nfsearch":4,
    #            "cluster":"fill","cluster_centroid_type":"noisy"}               
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v3,inds.kmb_v3,times.kmb_v3 = output
    # desc.kmb_v3 = "fill the cluster using noisy frames"

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ave_centroids","ave_centroid_type":"clean","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"clean"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v4,inds.kmb_v4,times.kmb_v4 = output

    # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"noisy","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v5,inds.kmb_v5,times.kmb_v5 = output

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"clean","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v1"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v6,inds.kmb_v6,times.kmb_v6 = output
    # desc.kmb_v6 = "Supervised clustering with a clean reference.\n"
    # desc.kmb_v6 += "The sup. clustering will not allow clusters of searched frames"

    # -- exec kmeans-burst search --

    # testing = {"ave":"ref_centroids","ave_centroid_type":"clean","nfsearch":3,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v7,inds.kmb_v7,times.kmb_v7 = output
    # desc.kmb_v7 = "Supervised clustering with a clean reference.\n"
    # desc.kmb_v7 += "This should be the best of all the clustering methods."

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"noisy","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v8,inds.kmb_v8,times.kmb_v8 = output
    # desc.kmb_v8 = "Supervised clustering with a single noisy reference.\n"
    # desc.kmb_v8 += "This is the standard search + known clusters."

    # -- exec kmeans-burst search --

    testing = {"ave":"ref_centroids","ave_centroid_type":"given","nfsearch":3,
               "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
               "sup_km_version":"v2"}
    output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    vals.kmb_v9,inds.kmb_v9,times.kmb_v9 = output
    desc.kmb_v9 = "Supervised clustering with a clustered noisy reference.\n"
    desc.kmb_v9 += "This is the known clusters + averaged noisy reference"

    testing = {"ave":"ref_centroids","ave_centroid_type":"clean","nfsearch":3,
               "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
               "sup_km_version":"v2"}
    output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    vals.kmb_v13,inds.kmb_v13,times.kmb_v13 = output
    desc.kmb_v13 = "Supervised clustering with a clean+noise reference.\n"

    testing = {"ave":"ref_centroids","ave_centroid_type":"clean-v1","nfsearch":3,
               "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
               "sup_km_version":"v2"}
    output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    vals.kmb_v14,inds.kmb_v14,times.kmb_v14 = output
    desc.kmb_v14 = "Supervised clustering with a clean+noise reference.\n"


    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"noisy","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2","sranges_type":"l2"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v10,inds.kmb_v10,times.kmb_v10 = output
    # desc.kmb_v10 = "Same as v8 but use the output from the l2 as the search grid."

    # # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"given","nfsearch":4,
    #            "cluster":"sup_kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2","sranges_type":"l2"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v11,inds.kmb_v11,times.kmb_v11 = output
    # desc.kmb_v11 = "Same as v9 but use the output from the l2 as the search grid."

    # -- exec kmeans-burst search --
    # testing = {"ave":"ref_centroids","ave_centroid_type":"given","nfsearch":4,
    #            "cluster":"kmeans","cluster_centroid_type":"noisy",
    #            "sup_km_version":"v2","sranges_type":"zero"}
    # output = run_kmb_search(noisy,clean,ps,nsearch_xy,std,vals,inds,times,testing)
    # vals.kmb_v12,inds.kmb_v12,times.kmb_v12 = output
    # desc.kmb_v12 = "Same as v9 but using actual kmeans"

    # -- print report --
    msg = "TODO"
    msg += "\nWe want to a non-sup clustering to be closer to sup.. e.g. beat l2. [7v9]"
    msg += "\nThe ave_centroid_type = clean should WIN given but it loses rn."
    print(msg)
    print_report(vals,inds,times,noisy,clean,ps)

def run_nnf_search(noisy,clean,patchsize,nsearch_xy,std,
                        agg_vals,agg_inds,agg_times):

    # -- extract into --
    device = noisy.device
    c,t,h,w = noisy.shape

    # -- exec search --
    valMean = 0.#compute_mode_pairs(std/255.,c,patchsize)
    noisy_tc = rearrange(noisy,'c t h w -> t 1 c h w')
    start_time = time.perf_counter()    
    nnf_vals,nnf_locs = nnf_utils.runNnfBurst(noisy_tc, patchsize, nsearch_xy,
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

    return vals,inds,runtime


def run_kmb_search(noisy,clean,patchsize,nsearch_xy,std,
                   agg_vals,agg_inds,agg_times,testing):
    # -- extract into --
    device = noisy.device
    c,t,h,w = noisy.shape

    # -- exec search --
    gt_dist = 0.
    gt_info = {'indices':agg_inds.gt,'clean':clean}
    noisy_tc = rearrange(noisy,'c t h w -> t 1 c h w')
    start_time = time.perf_counter()    
    _vals,_locs = runKmSearch(noisy_tc, patchsize, nsearch_xy, k = 1,
                              std = std/255.,search_space=None,
                              ref=None,mode="python",gt_info=gt_info,
                              testing=testing)
    runtime = time.perf_counter() - start_time

    # -- post process data --
    vals = _vals[0]
    inds = rearrange(_locs,'1 1 t h w two -> t 1 h w 1 two')
    _locs = rearrange(_locs,'1 1 t h w two -> two t h w')
    inds = flow2pix(inds)
    inds = rearrange(inds,'t 1 h w 1 two -> two t h w')
    inds = torch.flip(inds,dims=(0,))

    # -- to device --
    vals = vals.to(device)
    inds = inds.to(device)

    return vals,inds,runtime

def print_report(vals,inds,times,noisy,clean,patchsize):

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
            warp = warp_burst_from_pix(clean,m_inds[:,:,None])
            warp = rearrange(warp,'1 c t h w -> t c h w')
            psnrs = images_to_psnrs_crop(warp,repimg,2*divUp(ps,2))
            print(f"-- [{method}] --")
            print(psnrs)
            save_image(f"tkmb_{method}.png",warp)
    
