
# -- python --
import tqdm,math
import numpy as np
from einops import rearrange,repeat

# -- project --
from pyutils import save_image,get_img_coords

# -- pytorch --
import torch
import torchvision.transforms.functional as tvF

# -- faiss --
import faiss
from warp_utils import locs2flow,flow2locs,pix2locs,pix2flow

# -- kmb functionality --
from .kmeans_impl import run_kmeans,run_ekmeans,sup_kmeans
from .compute_mode_impl import compute_mode
from .compute_burst_dists_impl import compute_burst_dists
from .compute_burst_dists_impl import compute_ecentroid_dists
from .compute_burst_dists_impl import compute_ecentroid_edists
from .centroid_update_impl import fill_ecentroids,fill_sframes_ecentroids
from .compute_ave_impl import compute_centroid_ave,compute_ecentroid_ave
from .topk_impl import kmb_topk_update,update_state,topk_torch

from .centroid_update_impl import update_ecentroids
from .cluster_update_impl import init_clusters,sup_clusters

from .utils import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,divUp
from .utils import initialize_indices,pick_fill_frames,get_gt_info
from .utils import get_optional_field

from .debug import get_optimal_search_index


def vprint(*args,**kwargs):
    VERBOSE = False
    if VERBOSE: print(*args,**kwargs)


def th_bincount(clusters,sframes):
    t,s,h,w = clusters.shape
    for hi in range(h):
        for wi in range(w):
            for si in range(s):
                sclust = clusters[sframes,si,wi,hi]
                bins = torch.bincount(sclust)
                mbin = torch.max(bins)
                assert mbin == 1,"all must be one!"

def run_kmb_python(res, burst, patchsize, nsearch, k,
                   kmeansK, std, ref, search_ranges,
                   nsearch_xy=3, nsiters=2, nfsearch=4,
                   gt_info=None,testing=None):
    
    """
    Why can't we get to 160 PSNR when using clustering???
    """

    # -- defaults --
    t,c,h,w = burst.shape
    if nsearch_xy is None: nsearch_xy = 3
    if nfsearch is None: nfsearch = 5
    if nsiters is None: nsiters = divUp(t,nfsearch)
    clean,indices_gt = get_gt_info(gt_info)
    nfsearch = 3
    nsiters = 2*(t-nfsearch+1)#1*divUp(t,nfsearch)
    nframes = t

    # -- get running params --
    ave_fxn = get_ave_function(testing)
    cluster_fxn = get_cluster_function(testing)

    # -- shape --
    device = burst.device
    burst = rearrange(burst,'t c h w -> c t h w')
    c,t,h,w = burst.shape
    kmeansK = t-1
    ps = patchsize
    Z_l2 = ps*ps*c
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    
    # -- outputs --
    outT = nfsearch
    vals = torch.ones(outT,k,h,w).to(device)*float("nan")#1000
    inds = coords[:,:,None].repeat(1,1,k,1,1).type(torch.long) # 2,t,k,h,w
    modes = torch.zeros(outT,k,h,w).to(device)

    # -- search ranges --
    if search_ranges is None:
        search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)

    # -- search frames --
    search_frames = tiled_search_frames(t,nfsearch,nsiters,ref).to(device)
    search_frames = search_frames.type(torch.long)
    # vprint(search_frames)
    # vprint("burst.shape: ",burst.shape)
    # vprint("search_ranges.shape: ",search_ranges.shape)
    vprint("-- search frames --")
    vprint(search_frames)
    vprint("-"*30)

    # -- set current -- 
    curr_indices = initialize_indices(coords,search_ranges,indices_gt)

    # -- search subsets "nisters" times --
    indices_match = []
    kSched = np.array([kmeansK,]*nsiters)
    assert len(kSched) >= nsiters,"k sched and nisters."
    assert np.all(kSched <= t),"all less than num frames."
    prev_sframes = search_frames[0].clone()
    for s_iter in tqdm.tqdm(range(nsiters)):#nsiters

        vprint("-"*30)
        vprint("-"*30)
        vprint(f"Iteration: {s_iter}")
        vprint("-"*30)
        vprint("-"*30)

        kmeansK = kSched[s_iter]
        sframes = search_frames[s_iter]
        vprint("sframes: ",sframes)

        # -- rename for clarity --
        noisy = burst

        # -- create mesh --
        indices = mesh_from_ranges(search_ranges,sframes,curr_indices,ref)
        indices = indices.to(device).type(torch.long)

        # -- choose subset of frames from burst --
        iframes,alpha = sframes,2.
        iframes = pick_fill_frames(sframes,nfsearch,t,alpha,s_iter,device)

        # -- cluster function --
        output = cluster_fxn(burst,clean,kmeansK,indices,inidices_gt,sframes,iframes,ps)
        centroids,clusters,sizes = output

        # -- ave function --
        ave = ave_fxn(clean,noisy,centroids,clusters,sizes)

        # -- run kmeans for testing --
        # kimg = burst if clean is None else clean
        kimg = burst
        # # centroids,sizes,ave = fill_ecentroids(burst,indices,ps,kmeansK)
        # fgrid,alpha = sframes,2.
        # fgrid = pick_fill_frames(sframes,nfsearch,t,alpha,s_iter,device)
        # centroids,sizes,ave,rcl = fill_sframes_ecentroids(kimg,indices,fgrid,ps)
        # clusters = None
        # ave = rcl
        
        # -- run kmeans --
        # kimg = burst if clean is None else clean
        # vprint(sframes)
        # kmeans_out = sup_kmeans(burst,clean,indices,indices_gt,sframes,ps)
        # pwd_mode = 0.#compute_mode(std,c,ps,type='pairs')
        # # kmeans_out = run_ekmeans(kimg,indices,kmeansK,ps,pwd_mode,niters=8)
        # km_dists,clusters,sizes,centroids,rcl = kmeans_out
        # ave = rcl

        # ave = compute_ecentroid_ave(centroids,sizes)
        # ave = compute_ecentroid_ave(centroids,sizes)
        # ave = torch.nansum(centroids*sizes[None,...,None,None],dim=1)/nframes

        # print(ave[0,10,8,9,0])
        # print(centroids[0,:,10,8,9,0])
        # print(clusters[:,10,8,9])
        # exit()
        
        # print(centroids.shape)
        # print(centroids[0,:,0,8,9].transpose(0,1))
        # print(ave[0,0,8,9])
        # exit()

        # print(sizes.shape)
        # exit()
        # vprint("[burst] max,min: ",burst.max(),burst.min())
        # vprint("[centroids] max,min: ",centroids.max(),centroids.min())
        # ave = torch.mean(centroids,dim=0)
        # centroids = torch.zeros_like(centroids)
        # ave = torch.zeros_like(ave)
        # print(sizes[:,0,8,7])
        # print(sizes.shape)
        # exit()

        # print(clusters.shape)
        # th_bincount(clusters,sframes)
        # print(sizes.shape)
        # exit()

        # -- [testing] ave --
        kimg = burst if clean is None else clean
        kimg = burst#clean
        nframes = burst.shape[1]
        tclusters,tsizes = init_clusters(nframes,nframes,indices.shape[2],h,w,device)
        tcentroids = update_ecentroids(kimg,indices,tclusters,tsizes,ps)
        if clusters is None: clusters = tclusters
        # ave = tcentroids[:,ref]

        # -- compute modes --
        cmodes,_ = compute_mode(std,c,ps,sizes,type='centroids')
        # cmodes /= 2
        # _mode = compute_mode(std,c,ps,t,type='burst')
        # vprint(_mode)
        # vprint(cmodes[0,0,0,0])
        # cmodes[...] = _mode*Z_l2
        # cmodes = torch.mean(cmodes,dim=0)
        # cmodes = torch.mean(cmodes,dim=0)*Z_l2
        # vprint(cmodes)
        cmodes = torch.zeros_like(cmodes)

        # -- compute difference --
        # vprint(torch.any(torch.isnan(centroids)))
        # vprint(torch.any(torch.isnan(ave)))
        # l2_vals = compute_ecentroid_dists(centroids,sizes,ave)
        l2_vals = compute_ecentroid_edists(centroids,sizes,ave,t)
        l2_vals /= Z_l2
        vprint("l2_vals.shape: ",l2_vals.shape)
        vprint("cmodes.shape: ",cmodes.shape)
        vprint("sizes.shape: ",sizes.shape)

        # vprint(cmodes[:,0,4,5])
        # vprint(l2_vals[:,:,8,9])
        # vprint("-- (9,6) [bad] --")
        # vprint(cmodes.shape,l2_vals.shape)
        # mvals = torch.mean(torch.abs(l2_vals-cmodes),dim=0)
        # isorted = torch.argsort(mvals[:,9,6])[:3]
        # vprint(isorted)
        # vprint(mvals[isorted,9,6])
        # vprint(indices[:,:,isorted,9,6].transpose(0,2))
        # vprint(indices[:,:,:,9,6].transpose(0,2))
        # vprint(indices_gt[:,:,9,6].transpose(0,1))
        # vprint(curr_indices[:,:,9,6].transpose(0,1))

        # vprint("-- (9,7) [good] --")
        # vprint(l2_vals[:,:,9,7])
        # vprint(indices[:,:,:,9,7].transpose(0,2))
        # vprint(indices_gt[:,:,9,7].transpose(0,1))
        # vprint(curr_indices[:,:,9,7].transpose(0,1))

        vprint("-- (5,1) [bad] --")
        sindex = get_optimal_search_index(indices,indices_gt,sframes)
        # mvals = torch.nansum(torch.abs(l2_vals-cmodes),dim=0)
        mvals = torch.nansum(torch.abs(l2_vals-cmodes)*sizes/nframes,dim=0)
        isorted = torch.argsort(mvals[:,5,1])[:3]

        # -- useful print info --
        vprint("mvals.shape: ",mvals.shape)
        vprint("clusters.shape: ",clusters.shape)
        vprint("Winner Should Be Index: ",sindex)
        vprint("Top 3 Winners Actually Are: ",isorted)
        vprint("-- cluster info [@idx_match,isort[0],isort...] --")
        vprint(clusters[:,sindex[0],5,1])
        vprint(clusters[:,isorted[0],5,1])
        vprint(clusters[:,isorted[1],5,1])
        vprint(clusters[:,isorted[2],5,1])

        # -- delete me --
        # vprint(clusters[:,45,5,1])
        # vprint(clusters[:,51,5,1])
        # vprint(clusters[:,79,5,1])
        # -- keep me --
        vprint(sframes)
        vprint(l2_vals[:,:4,5,1])
        # vprint(indices[:,-4:,:,5,1].transpose(0,2))
        vprint(" -- indices gt --")
        vprint(indices_gt[:,:,5,1])
        vprint(" -- current indices --")
        vprint(curr_indices[:,:,5,1])

        vprint(" ------------- ")
        vprint(" -- sindex -- ")
        vprint(" ------------- ")
        vprint(sindex)
        vprint(" -- vals[:,@match,5,1] --")
        vprint(l2_vals[:,sindex[0],5,1])
        vprint(" -- modes[:,@match,5,1] --")
        vprint(cmodes[:,sindex[0],5,1])
        vprint(" -- sizes[:,@match,5,1] --")
        vprint(sizes[:,sindex[0],5,1])
        vprint(" -- mvals[@match,5,1] --")
        vprint(mvals[sindex[0],5,1])
        vprint(" -- indices[:,:,@match,5,1] --")
        vprint(indices[:,:,sindex,5,1].transpose(1,2).transpose(0,1))

        vprint(" ------------- ")
        vprint(" -- isorted -- ")
        vprint(" ------------- ")
        vprint(isorted)
        vprint(" -- vals[:,@isorted[0],5,1] --")
        vprint(l2_vals[:,isorted[0],5,1])
        vprint(" -- modes[:,@isorted[0],5,1] --")
        vprint(cmodes[:,isorted[0],5,1])
        vprint(" -- sizes[:,@isorted[0],5,1] --")
        vprint(sizes[:,isorted[0],5,1])
        vprint(" -- mvals[isorted,5,1] --")
        vprint(mvals[isorted,5,1])
        vprint(" -- indices[:,:,isorted,5,1] --")
        vprint(indices[:,:,isorted,5,1].transpose(1,2).transpose(0,1))
        

        # -- create top k --
        vprint("[create top k] indices.shape: ",indices.shape)
        # vprint(inds.shape)
        # vprint(cmodes.shape)
        # vprint(modes.shape)
        # vals,modes,inds = update_state(l2_vals,vals,cmodes,modes,
        #                                indices,inds,sframes,s_iter)
        # mvals = torch.nansum(torch.abs(l2_vals - cmodes),dim=0)
        mvals = torch.nansum(torch.abs(l2_vals-cmodes)*sizes/nframes,dim=0)
        vprint("[creat top k] mvals.shape: ",mvals.shape)
        isorted = torch.argsort(mvals[:,5,1])[:3]
        vprint("[create top k] argsort(mvals): ", isorted)
        cmodes_ave = torch.mean(cmodes,dim=1)[:,None]
        vals,modes,inds = topk_torch(mvals,vals,cmodes_ave,indices,1)
        # inds = inds[:,:,0]
        # vals,modes,inds = kmb_topk_update(l2_vals,vals,
        #                                   cmodes,modes,
        #                                   indices,inds,
        #                                   sframes,prev_sframes)
        prev_sframes = sframes.clone()
        # vals,modes,inds = kmb_topk_update(l2_vals,vals,cmodes,modes,indices,inds)
        # vprint(l2_vals)
        vprint("vals.shape: ",vals.shape)
        # vprint(inds)
        vprint("inds.shape: ",inds.shape)
        delta = torch.abs(inds[:,:,0] - curr_indices)
        delta = torch.mean(delta.type(torch.float)).item()
        # vprint("Delta: ",delta)
        curr_indices = inds[:,:,0]
        vprint("-- [bottom] curr indices.")
        vprint(curr_indices[:,:,5,1])
        vprint("-- [bottom] gt indices.")
        vprint(indices_gt[:,:,5,1])
        # if s_iter == 1: exit()

        # -- print the shit ones --
        shit_indices = curr_indices != indices_gt
        shit_indices = torch.any(shit_indices,dim=0)
        shit_indices = torch.where(shit_indices)
        shit_indices = torch.stack(shit_indices,dim=-1)
        # vprint(" -- the shit ones. --")
        # vprint("curr_indices.shape: ",curr_indices.shape)
        # vprint("indices_gt.shape: ",indices_gt.shape)
        # vprint("shit_indices.shape: ",shit_indices.shape)
        # vprint(shit_indices[torch.where(shit_indices[:,0] == 0)])

        # vprint("curr_indices.shape: ",curr_indices.shape)
        indices_match_s = (curr_indices == indices_gt)
        indices_match_s = torch.all(indices_match_s,dim=0)
        indices_match_s = indices_match_s.type(torch.float)
        # vprint("indices_match.shape: ",indices_match_s.shape)
        indices_match.append(indices_match_s)
        # if s_iter == 0: exit()

    indices_match = torch.stack(indices_match,dim=0)[:,:,None]
    for t in range(indices_match.shape[1]):
        save_image(f"tkmb_indices_{t}.png",indices_match[:,t])

    # vprint("-- vals --")
    # vprint(vals.shape)
    # vprint(vals[:,4,4])
    # vprint("-- inds --")
    # vprint(inds.shape)
    # vprint(inds[:,:,0,4,4])

    # -- swap dim --
    def swap_2d_dim(tensor,dim):
        tensor = tensor.clone()
        tmp = tensor[0].clone()
        tensor[0] = tensor[1].clone()
        tensor[1] = tmp
        return tensor

    # -- convert to pix --
    inds = curr_indices[:,:,None]
    inds = swap_2d_dim(inds,0)
    inds = rearrange(inds,'two t k h w -> t 1 h w k two')
    locs = pix2flow(inds)
    locs = rearrange(locs,'t 1 h w k two -> two t k h w')

    return vals,locs,modes
