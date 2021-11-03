
# -- python --
import tqdm
import numpy as np
from einops import rearrange,repeat

# -- project --
from pyutils import save_image,get_img_coords

# -- pytorch --
import torch
import torchvision.transforms.functional as tvF

# -- faiss --
import faiss
from warp_utils import locs2flow,flow2locs,pix2locs

# -- kmb functionality --
from .kmeans_impl import run_kmeans,run_ekmeans
from .compute_mode_impl import compute_mode
from .compute_burst_dists_impl import compute_burst_dists
from .compute_burst_dists_impl import compute_ecentroid_dists
from .compute_burst_dists_impl import compute_ecentroid_edists
from .centroid_update_impl import fill_ecentroids,fill_sframes_ecentroids
from .compute_ave_impl import compute_centroid_ave,compute_ecentroid_ave
from .topk_impl import kmb_topk_update,update_state
from .utils import jitter_search_ranges,tiled_search_frames,mesh_from_ranges
def divUp(a,b): return (a-1)//b+1


def run_kmb_python(res, burst , patchsize, nsearch, k,
                   kmeansK, std, ref, search_ranges,
                   nsearch_xy=3, nsiters=2, nfsearch=4,
                   gt_info=None):
    
    # -- defaults --
    t,c,h,w = burst.shape
    if nsearch_xy is None: nsearch_xy = 3
    if nfsearch is None: nfsearch = 5
    if nsiters is None: nsiters = divUp(t,nfsearch)
    if not(gt_info is None): indices_gt = gt_info['indices']
    else: indices_gt = None
    nfsearch = 2
    nsiters = 10

    # -- shape --
    device = burst.device
    t,c,h,w = burst.shape
    kmeansK = nfsearch
    burst = rearrange(burst,'t c h w -> c t h w')
    ps = patchsize
    Z_l2 = ps*ps*c
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    
    # -- outputs --
    vals = torch.ones(t,k,h,w).to(device)*float("nan")#1000
    inds = torch.zeros(2,t,k,h,w).to(device).type(torch.long)
    modes = torch.zeros(t,k,h,w).to(device)

    # -- search ranges --
    if search_ranges is None:
        search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)

    # -- search frames --
    search_frames = tiled_search_frames(t,nfsearch,nsiters,ref).to(device)
    search_frames = search_frames.type(torch.long)
    # print(search_frames)
    # print("burst.shape: ",burst.shape)
    # print("search_ranges.shape: ",search_ranges.shape)
    print("-- search frames --")
    print(search_frames)
    print("-"*30)

    # -- set current -- 
    inds = coords[:,:,None].repeat(1,1,k,1,1).type(torch.long)
    # rinit = search_ranges.shape[2]//2
    rmax = search_ranges.shape[2]
    rinit = torch.randint(0,rmax,(t,1,h,w)).to(device)
    # print(rinit.shape,search_ranges.shape)
    # curr_indices = torch.zeros_like(search_ranges)
    curr_indices = search_ranges[:,:,0]
    # curr_indices = coords.clone()
    # print("curr_indices.shape: ",curr_indices.shape)
    # coords.clone() --> (2,t,h,w)
    # curr_indices[0] = torch.gather(search_ranges[0],dim=1,index=rinit)
    # curr_indices[1] = torch.gather(search_ranges[1],dim=1,index=rinit)
    # curr_indices = curr_indices.type(torch.long)
    # curr_indices = curr_indices[:,:,0]
    # curr_indices = indices_gt.clone()

    # -- search subsets "nisters" times --
    indices_match = []
    kSched = np.array([kmeansK,]*nsiters)
    assert len(kSched) >= nsiters,"k sched and nisters."
    assert np.all(kSched <= t),"all less than num frames."
    prev_sframes = search_frames[0].clone()
    for s_iter in tqdm.tqdm(range(nsiters)):#nsiters

        # print(f"Iteration: {s}")
        # print(search_frames[s])
        kmeansK = kSched[s_iter]
        sframes = search_frames[s_iter]
        print(sframes)
        # -- create mesh --
        indices = mesh_from_ranges(search_ranges,sframes,curr_indices,ref)
        indices = indices.to(device).type(torch.long)
        # print(indices.shape)
        # print(indices[:,:,:,9,6].transpose(0,-1))
        # print(curr_indices[:,:,9,6])
        # indices = torch.zeros_like(indices)
        # if s_iter == 1: exit()

        # -- run kmeans for testing --
        # centroids,sizes,ave = fill_ecentroids(burst,indices,ps,kmeansK)
        centroids,sizes,ave = fill_sframes_ecentroids(burst,indices,sframes,ps)

        # -- run kmeans --
        # pwd_mode = 0.#compute_mode(std,c,ps,type='pairs')
        # kmeans_out = run_ekmeans(burst,indices,kmeansK,ps,pwd_mode,niters=5)
        # km_dists,clusters,sizes,centroids = kmeans_out
        # ave = compute_ecentroid_ave(centroids,sizes)
        # print("[burst] max,min: ",burst.max(),burst.min())
        # print("[centroids] max,min: ",centroids.max(),centroids.min())
        # ave = torch.mean(centroids,dim=0)
        # centroids = torch.zeros_like(centroids)
        # ave = torch.zeros_like(ave)

        # -- compute modes --
        cmodes,_ = compute_mode(std,c,ps,sizes,type='centroids')
        # cmodes /= 2
        # _mode = compute_mode(std,c,ps,t,type='burst')
        # print(_mode)
        # print(cmodes[0,0,0,0])
        # cmodes[...] = _mode*Z_l2
        # cmodes = torch.mean(cmodes,dim=0)
        # cmodes = torch.mean(cmodes,dim=0)*Z_l2
        # print(cmodes)
        cmodes = torch.zeros_like(cmodes)

        # -- compute difference --
        # print(torch.any(torch.isnan(centroids)))
        # print(torch.any(torch.isnan(ave)))
        # l2_vals = compute_ecentroid_dists(centroids,sizes,ave)
        l2_vals = compute_ecentroid_edists(centroids,sizes,ave,t)
        l2_vals /= Z_l2
        # print("l2_vals.shape: ",l2_vals.shape)
        # print(cmodes[:,0,4,5])
        # print(l2_vals[:,:,8,9])
        # print("-- (9,6) [bad] --")
        # print(cmodes.shape,l2_vals.shape)
        # mvals = torch.mean(torch.abs(l2_vals-cmodes),dim=0)
        # isorted = torch.argsort(mvals[:,9,6])[:3]
        # print(isorted)
        # print(mvals[isorted,9,6])
        # print(indices[:,:,isorted,9,6].transpose(0,2))
        # print(indices[:,:,:,9,6].transpose(0,2))
        # print(indices_gt[:,:,9,6].transpose(0,1))
        # print(curr_indices[:,:,9,6].transpose(0,1))

        # print("-- (9,7) [good] --")
        # print(l2_vals[:,:,9,7])
        # print(indices[:,:,:,9,7].transpose(0,2))
        # print(indices_gt[:,:,9,7].transpose(0,1))
        # print(curr_indices[:,:,9,7].transpose(0,1))

        print("-- (4,5) [bad] --")
        # print(sframes)
        # print(l2_vals[:,-4:,4,5])
        # print(indices[:,-4:,:,4,5].transpose(0,2))
        print(indices_gt[:,:,4,5].transpose(0,1))
        # print(curr_indices[:,:,4,5].transpose(0,1))

        mvals = torch.mean(torch.abs(l2_vals-cmodes),dim=0)
        isorted = torch.argsort(mvals[:,4,5])[:3]
        print(isorted)
        print(mvals[isorted,4,5])
        print(indices[:,:,isorted,4,5].transpose(0,2))


        # -- create top k --
        # print(indices.shape)
        # print(inds.shape)
        # print(cmodes.shape)
        # print(modes.shape)
        # vals,modes,inds = update_state(l2_vals,vals,cmodes,modes,
        #                                indices,inds,sframes,s_iter)
        vals,modes,inds = kmb_topk_update(l2_vals,vals,
                                          cmodes,modes,
                                          indices,inds,
                                          sframes,prev_sframes)
        prev_sframes = sframes.clone()
        # vals,modes,inds = kmb_topk_update(l2_vals,vals,cmodes,modes,indices,inds)
        # print(l2_vals)
        # print("vals.shape: ",vals.shape)
        # print(inds)
        # print("inds.shape: ",inds.shape)
        delta = torch.abs(inds[:,:,0] - curr_indices)
        delta = torch.mean(delta.type(torch.float)).item()
        # print("Delta: ",delta)
        curr_indices = inds[:,:,0]
        # print("-- curr indices.")
        # print(curr_indices[:,:,9,6].transpose(0,1))
        # print("-- gt indices.")
        # print(indices_gt[:,:,9,6].transpose(0,1))

        # -- print the shit ones --
        indices_match_s = curr_indices != indices_gt
        indices_match_s = torch.any(indices_match_s,dim=0)
        indices_match_s = torch.where(indices_match_s)
        # print(len(indices_match_s))
        indices_match_s = torch.stack(indices_match_s,dim=-1)
        # print(indices_match_s.shape)
        # print(indices_match_s[300:400])
        # print(indices_match_s[2])
        # print(indices_match_s[3])

        # print("curr_indices.shape: ",curr_indices.shape)
        indices_match_s = (curr_indices == indices_gt)
        indices_match_s = torch.all(indices_match_s,dim=0)
        indices_match_s = indices_match_s.type(torch.float)
        # print("indices_match.shape: ",indices_match_s.shape)
        indices_match.append(indices_match_s)
        if s_iter > 1: exit()

    indices_match = torch.stack(indices_match,dim=0)[:,:,None]
    for t in range(indices_match.shape[1]):
        save_image(f"tkmb_indices_{t}.png",indices_match[:,t])

    # print("-- vals --")
    # print(vals.shape)
    # print(vals[:,4,4])
    # print("-- inds --")
    # print(inds.shape)
    # print(inds[:,:,0,4,4])

    # -- swap dim --
    def swap_2d_dim(tensor,dim):
        tensor = tensor.clone()
        tmp = tensor[0].clone()
        tensor[0] = tensor[1].clone()
        tensor[1] = tmp
        return tensor

    # -- convert to pix --
    inds = swap_2d_dim(inds,0)
    inds = rearrange(inds,'two t k h w -> t 1 h w k two')
    locs = pix2locs(inds)
    locs = rearrange(locs,'t 1 h w k two -> two t k h w')

    return vals,locs,modes
