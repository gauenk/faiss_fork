
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
from .parse_utils import get_centroids_for_ave

from .debug import get_optimal_search_index

# -- load functions --
from .parse_cluster_impl import get_cluster_function
from .parse_ave_impl import get_ave_function
from .parse_mode_impl import get_mode_function
from .parse_score_impl import get_score_function


VERBOSE = False
def vprint(*args,**kwargs):
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

def run_kmb_python(res, noisy, patchsize, nsearch, k,
                   kmeansK, std, ref, search_ranges,
                   nsearch_xy=3, nsiters=2, nfsearch=4,
                   gt_info=None,testing=None):
    
    """
    Why can't we get to 160 PSNR when using clustering???
    """

    # -- defaults --
    global VERBOSE
    t,c,h,w = noisy.shape
    if nsearch_xy is None: nsearch_xy = 3
    if nfsearch is None: nfsearch = 5
    if nsiters is None: nsiters = divUp(t,nfsearch)
    clean,indices_gt = get_gt_info(gt_info)
    nfsearch = get_optional_field(testing,"nfsearch",3)
    nsiters = 3*(t-nfsearch+1)#1*divUp(t,nfsearch)
    nsiters = get_optional_field(testing,"nsiters",nsiters)
    VERBOSE = get_optional_field(testing,"verbose",VERBOSE)
    nframes = t

    # -- get running params --
    ave_fxn = get_ave_function(testing)
    cluster_fxn = get_cluster_function(testing)
    mode_fxn = get_mode_function(testing)
    score_fxn = get_score_function(testing)

    # -- shape --
    device = noisy.device
    noisy = rearrange(noisy,'t c h w -> c t h w')
    c,t,h,w = noisy.shape
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
    # vprint("noisy.shape: ",noisy.shape)
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

        # if s_iter == 0: VERBOSE = True
        # else: VERBOSE = False
        vprint("-"*30)
        vprint("-"*30)
        vprint(f"Iteration: {s_iter}")
        vprint("-"*30)
        vprint("-"*30)

        kmeansK = kSched[s_iter]
        sframes = search_frames[s_iter]
        vprint("sframes: ",sframes)

        # -- create mesh --
        indices = mesh_from_ranges(search_ranges,sframes,curr_indices,ref)
        indices = indices.to(device).type(torch.long)

        # -- choose subset of frames from noisy --
        iframes,alpha = sframes,2.
        # iframes = pick_fill_frames(sframes,nfsearch,t,alpha,s_iter,device)

        # -- cluster function --
        output = cluster_fxn(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps)
        centroids,clusters,sizes = output

        # -- ave function --
        ave = ave_fxn(noisy,clean,centroids,clusters,sizes,indices,ps)

        # # -- [testing] ave --
        # # kimg = noisy if clean is None else clean
        # kimg = noisy#clean
        # nframes = noisy.shape[1]
        # tclusters,tsizes = init_clusters(nframes,nframes,indices.shape[2],h,w,device)
        # tcentroids = update_ecentroids(kimg,indices,tclusters,tsizes,ps)
        # if clusters is None: clusters = tclusters
        # ave = tcentroids[:,ref]

        # -- mode function --
        cmodes = mode_fxn(std,c,ps,sizes)
        
        # -- compute distance --
        l2_vals = compute_ecentroid_edists(centroids,sizes,ave,nframes)
        l2_vals /= Z_l2
        vprint("l2_vals.shape: ",l2_vals.shape)
        vprint("cmodes.shape: ",cmodes.shape)
        vprint("sizes.shape: ",sizes.shape)

        # -- compute score for ranking --
        scores = score_fxn(l2_vals,cmodes,clusters,sizes,nframes,ref)
        mvals = scores

        #  -------------------------
        #
        #        OLD STUFF
        #
        #  -------------------------
        
        # -- compute difference --
        # vprint(torch.any(torch.isnan(centroids)))
        # vprint(torch.any(torch.isnan(ave)))
        # l2_vals = compute_ecentroid_dists(centroids,sizes,ave)

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

        vprint("-- (5,8) [bad] --")
        sindex = get_optimal_search_index(indices,indices_gt,sframes)
        # mvals = torch.nansum(torch.abs(l2_vals-cmodes),dim=0)
        # mvals = torch.nansum(torch.abs(l2_vals-cmodes)*sizes/nframes,dim=0)
        isorted = torch.argsort(mvals[:,5,8])[:3]

        # -- useful print info --
        vprint("mvals.shape: ",mvals.shape)
        vprint("clusters.shape: ",clusters.shape)
        vprint("Winner Should Be Index: ",sindex)
        vprint("Top 3 Winners Actually Are: ",isorted)

        # -- centroids --
        # clean_centroids,_,_ = get_centroids_for_ave("clean",noisy,clean,indices,ps)
        # noisy_centroids,_,_ = get_centroids_for_ave("noisy",noisy,clean,indices,ps)
        # vprint(" -- [clean] centroids[0,:,324,5,8] --")
        # vprint(clean_centroids[0,:,324,5,8])
        # vprint(" -- [noisy] centroids[0,:,324,5,8] --")
        # vprint(noisy_centroids[0,:,324,5,8])
        # vprint(" -- centroids[0,:,324,5,8] --")
        # vprint(centroids[0,:,324,5,8])

        # vprint(" -- [clean] centroids[0,:,326,5,8] --")
        # vprint(clean_centroids[0,:,326,5,8])
        # vprint(" -- [noisy] centroids[0,:,326,5,8] --")
        # vprint(noisy_centroids[0,:,326,5,8])
        # vprint(" -- centroids[0,:,326,5,8] --")
        # vprint(centroids[0,:,326,5,8])
        



        # -- delete me --
        # vprint(clusters[:,45,5,8])
        # vprint(clusters[:,51,5,8])
        # vprint(clusters[:,79,5,8])
        # -- keep me --
        vprint(sframes)
        vprint(l2_vals[:,:4,5,8])
        # vprint(indices[:,-4:,:,5,8].transpose(0,2))
        vprint(" -- indices gt --")
        vprint(indices_gt[:,:,5,8])
        vprint(" -- current indices --")
        vprint(curr_indices[:,:,5,8])

        vprint(" ------------- ")
        vprint(" -- sindex -- ")
        vprint(" ------------- ")
        vprint(sindex)
        vprint(" -- vals[:,@match,5,8] --")
        vprint(l2_vals[:,sindex[0],5,8])
        vprint(" -- modes[:,@match,5,8] --")
        vprint(cmodes[:,sindex[0],5,8])
        vprint(" -- sizes[:,@match,5,8] --")
        vprint(sizes[:,sindex[0],5,8])
        vprint(" -- mvals[@match,5,8] --")
        vprint(mvals[sindex[0],5,8])
        vprint(" -- indices[:,:,@match,5,8] --")
        vprint(indices[:,:,sindex,5,8].transpose(1,2).transpose(0,1))
        vprint("-- clusters[:,@match,5,8] --")
        vprint(clusters[:,sindex[0],5,8])

        vprint(" ------------- ")
        vprint(" -- isorted -- ")
        vprint(" ------------- ")
        vprint(isorted)
        vprint(" -- vals[:,@isorted[0],5,8] --")
        vprint(l2_vals[:,isorted[0],5,8])
        vprint(l2_vals[:,isorted[1],5,8])
        vprint(" -- modes[:,@isorted[0],5,8] --")
        vprint(cmodes[:,isorted[0],5,8])
        vprint(cmodes[:,isorted[1],5,8])
        vprint(" -- sizes[:,@isorted[0],5,8] --")
        vprint(sizes[:,isorted[0],5,8])
        vprint(sizes[:,isorted[1],5,8])
        vprint(" -- mvals[isorted,5,8] --")
        vprint(mvals[isorted,5,8])
        vprint(" -- indices[:,:,isorted,5,8] --")
        vprint(indices[:,:,isorted,5,8].transpose(1,2).transpose(0,1))
        vprint(" -- clusters[:,isorted,5,8] --")
        vprint(clusters[:,isorted[0],5,8])
        vprint(clusters[:,isorted[1],5,8])
        vprint(clusters[:,isorted[2],5,8])

        # vprint(" ------------- ")
        # vprint(" -- @324 -- ")
        # vprint(" ------------- ")
        # vprint(" -- vals[:,@324,5,8] --")
        # vprint(l2_vals[:,324,5,8])
        # vprint(" -- modes[:,@324,5,8] --")
        # vprint(cmodes[:,324,5,8])
        # vprint(" -- sizes[:,@324,5,8] --")
        # vprint(sizes[:,324,5,8])
        # vprint(" -- mvals[@324,5,8] --")
        # vprint(mvals[324,5,8])
        # vprint(" -- indices[:,:,@324,5,8] --")
        # vprint(indices[:,:,324,5,8])
        # vprint(" -- clusters[:,@324,5,8] --")
        # vprint(clusters[:,324,5,8])

        
        # -- create top k --
        vprint("[create top k] indices.shape: ",indices.shape)
        # vprint(inds.shape)
        # vprint(cmodes.shape)
        # vprint(modes.shape)
        # vals,modes,inds = update_state(l2_vals,vals,cmodes,modes,
        #                                indices,inds,sframes,s_iter)
        # mvals = torch.nansum(torch.abs(l2_vals - cmodes),dim=0)
        # mvals = torch.nansum(torch.abs(l2_vals-cmodes)*sizes/nframes,dim=0)
        vprint("[creat top k] mvals.shape: ",mvals.shape)
        isorted = torch.argsort(mvals[:,5,8])[:3]
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
        vprint(curr_indices[:,:,5,8])
        vprint("-- [bottom] gt indices.")
        vprint(indices_gt[:,:,5,8])
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
        shit_indices = shit_indices[torch.where(shit_indices[:,0] == 0)]
        vprint(shit_indices.transpose(0,1))

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
