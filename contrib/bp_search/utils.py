

import torch
import torchvision
import numpy as np
from einops import rearrange,repeat
from nnf_share import getBlockLabelsRaw,loc_index_names,pix2locs,warp_burst_from_pix,warp_burst_from_locs
from align.xforms import align_from_pix
from .th_kmeans import KMeans
from .meshgrid_per_pixel import meshgrid_per_pixel_launcher
center_crop = torchvision.transforms.functional.center_crop

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

def batched_scatter_index(x, val, nftrs):
    target = torch.zeros(x.shape[0], x.shape[1], nftrs, dtype=x.dtype, device=x.device)
    target.scatter_add_(2, x, val)
    return target

def create_search_ranges(nblocks,h,w,nframes,ref=None):

    # -- search range per pixel --
    if ref is None: ref = nframes//2
    search_ranges = getBlockLabelsRaw(nblocks) 
    search_ranges = repeat(search_ranges,'l two -> l h w t two',h=h,w=w,t=nframes)
    search_ranges[...,ref,:] = 0
    return search_ranges

def compute_search_blocks(sranges,refG,img_shape=None):

    # -- init shapes and smesh -- 
    # print(sranges.shape)
    nsearch_per_group,nimages,h,w,ngroups,two = sranges.shape
    # print("ngroups: ",ngroups)
    nsearch = nsearch_per_group**(ngroups-1) # since ref only has 1 elem
    # smesh = torch.zeros(nsearch,nimages,h,w,ngroups,two).type(torch.int)
    # smesh = smesh.to(sranges.device)

    # # -- numba-fy values --
    # for i in range(nimages):
    #     meshgrid_per_pixel_launcher(sranges[:,i],smesh[:,i],refG)
    #     smesh[:,i] = torch.flip(smesh[:,i],dims=(-1,))

    # -- [for testing] version 2 --
    ranges0 = sranges[:,0,0,0,-1,0].cpu().numpy()
    ranges1 = sranges[:,0,0,0,-1,1].cpu().numpy()
    ranges0 = repeat(ranges0,'s -> s g',g=ngroups)
    ranges1 = repeat(ranges1,'s -> s g',g=ngroups)
    # ranges0[refG] = [0]
    # ranges1[refG] = [0]
    # print("ranges0.shape: ",ranges0.shape)

    ranges0 = [ ranges0[:,s] for s in range(ngroups)]
    ranges1 = [ ranges1[:,s] for s in range(ngroups)]
    # print(ranges0[0])
    # print("-"*30)
    ranges0[refG] = np.array([0])
    ranges1[refG] = np.array([0])
    # print(len(ranges0))
    # for s in range(ngroups): print(s,len(ranges0[s]))

    smesh0 = np.meshgrid(*ranges0)
    smesh0 = np.stack([g.ravel() for g in smesh0])
    smesh1 = np.meshgrid(*ranges1)
    smesh1 = np.stack([g.ravel() for g in smesh1])

    # print("smesh0.shape: ",smesh0.shape)
    smesh = np.stack([smesh0,smesh1],axis=-1)
    # print("smesh.shape: ",smesh.shape)
    smesh = repeat(smesh,'g l two -> l i h w g two',i=nimages,h=h,w=w)
    # print("smesh.shape: ",smesh.shape)
    smesh = torch.IntTensor(smesh).to(sranges.device)
    # print("nsearch: ",nsearch)

    # -- crop to img_shape --
    if not(img_shape is None):
        smesh = rearrange(smesh,'l i h w t two -> l i t two h w')
        smesh = center_crop(smesh,img_shape[1:])
        smesh = rearrange(smesh,'l i t two h w -> l i h w t two')
        smesh = smesh.contiguous()

    return smesh
    
def add_offset_to_search_ranges(locs,search_ranges):
    
    # -- unpack shapes
    nframes,h,w,k,two = locs.shape
    nparticles = k

    # -- create offsets --
    search_ranges_all = []
    for p in range(nparticles):
        search_ranges_all[p] = locs[:,:,p,:] + search_ranges
    search_ranges_all = torch.stack(search_ranges_all)

    return search_ranges_all

def temporal_inliers_outliers(burst,wburst,vals,std,numSearch=3,ref=None):

    # -- shapes --
    device = burst.device
    nframes,nimages,nftrs,h,w = wburst.shape
    if ref is None: ref = nframes//2
    minNumOutliers = numSearch
    maxNumNeigh = nframes - minNumOutliers
    # maxNumNeigh = 3
    # maxNumNeigh = 12

    # -- compute threshold --
    # thresh = std * (nframes - 1) / nframes**2
    # thresh = 0.108#std * (nframes - 1) / nframes**2
    # thresh = std*255.*.105+.5
    thresh = 100.#0.75 #+ 1.
    # print("thresh: ",thresh)#,std,std*255.,std*255.*.105,nftrs)

    # -- compute per-pixel assignments --
    burst_rs = rearrange(burst,'t i f h w -> t i h w f')
    wburst_rs = rearrange(wburst,'t i f h w -> t i h w f')
    wburst_fmt = rearrange(wburst,'t i f h w -> (i h w) t f')

    # -- compute pair-wise distances --
    B,T,F = wburst_fmt.shape
    D = torch.cdist(wburst_fmt,wburst_fmt)[:,ref]

    # -- reshape back to image --
    D = rearrange(D,'(i h w) t -> t i h w',i=nimages,h=h,w=w)
    # print(D[:,0,16+1,16+1])
    jitter = torch.rand(D.shape).to(device)/2.


    # # -- filter by threshold -- 
    # D = torch.where(D < thresh, D.double(), torch.finfo(D.dtype).max)

    # -- sorted Distances and Neighbors --
    order = torch.argsort(D,dim=0)
    bestOrder = order[:maxNumNeigh]
    bestD = torch.gather(D,0,bestOrder)

    #
    # -- mark outlier images -- 
    #

    # -- mark each frame (h,w) if an inlier --
    counts = torch.zeros_like(order)
    ones = torch.ones_like(order)
    counts.scatter_(0,bestOrder,ones) # if "bestOrder" picked you, label with a "1"
    counts = torch.where(D < thresh,counts,0) # remove if the dist is too big.
    
    # -- compute percent of inliers pixels per frame -- 
    perc_pix_inliers = torch.mean(counts.type(torch.float),dim=(2,3))
    # jitter = torch.rand(perc_pix_inliers.shape).to(device)/100.
    # jitter = torch.zeros_like(jitter)
    # perc_pix_inliers += jitter # -- we don't want matches --

    # -- mark frames as outliers -- 
    percThresh = torch.sort(perc_pix_inliers,
                            dim=0,descending=True).values[maxNumNeigh].item()
    percThresh = max(percThresh,0.90)
    outliers = torch.where(perc_pix_inliers < percThresh)[0]
    # print(percThresh,outliers)
    assert len(outliers) >= minNumOutliers,"Min num of outliers met."
    
    #
    # -- choose outliers for search --
    # 

    noutliers = len(outliers)
    search_names_indices = torch.randperm(noutliers)[:numSearch]
    search_names = outliers[search_names_indices]
    search_frames = burst_rs[search_names]

    #
    # -- compute average over inliers pixels _without_ outliers --
    # 

    # -- remove chosen outliers --
    for outlier_fname in search_names:
        bestOrder = torch.where(bestOrder == outlier_fname,ref,bestOrder)

    # -- compute num uniques per pixel --
    bestOrder_rs = rearrange(bestOrder,'t i h w -> (i h w) t').contiguous()
    fcounts = batched_bincount(bestOrder_rs,1,nframes)
    fcounts = rearrange(fcounts,'(i h w) t -> t i h w',h=h,w=w)
    inlier_mask = torch.where(fcounts >= 1,1,0)
    nuniques = torch.sum(inlier_mask,dim=0)
    assert torch.any(nuniques > maxNumNeigh) == False, "Limit max search."

    # -- gather inlier frames --
    sub_fcounts = torch.zeros_like(bestOrder).type(fcounts.dtype)
    torch.gather(fcounts,0,bestOrder,out=sub_fcounts)
    bestOrder = bestOrder[...,None].expand(-1,-1,-1,-1,F)
    inliers = torch.zeros_like(bestOrder).type(wburst_rs.dtype)
    torch.gather(wburst_rs,0,bestOrder,out=inliers)
    
    # -- reweight by individual frame contrib --
    fcounts_exp = repeat(sub_fcounts,'t i h w -> t i h w f',f=F)
    nuniques_exp = repeat(nuniques,'i h w -> i h w f',f=F)
    winliers = inliers / fcounts_exp # re-scale over-represented samples
    inlier_ave = torch.sum(winliers,dim=0) / nuniques_exp # correct divisor for ave

    #
    # -- reformat & stack images --
    #

    inlier_ave = rearrange(inlier_ave,'i h w f -> 1 i f h w')
    search_frames = rearrange(search_frames,'t i h w f -> t i f h w')
    search_frames = torch.cat([inlier_ave,search_frames],dim=0)
    nuniques = nuniques + search_frames.shape[0] - 1

    return search_frames,search_names,nuniques

def flow_to_groups(flow):
    # i t h w two

    # -- reshape for readability --
    nimages,nframes,H,W,two = flow.shape
    flow = rearrange(flow,'i t h w two -> (i h w) t two')
    B,T,two = flow.shape

    # # -- group flows --
    # groups = torch.zeros(nimages,H,W,ngroups).astype(torch.bool)

    # -- get pairwise bools --
    eq_flow = torch.zeros(nframes,nframes,B).type(torch.bool)
    for t0 in range(nframes):
        for t1 in range(nframes):
            eq_flow[t0,t1] = torch.all(flow[:,t0] == flow[:,t1],dim=-1)

    
    # -- from bool matrix of frames to pairwise frames --
    groups = torch.zeros(B,nframes).type(torch.long)
    for t0 in range(nframes):    
        _,indices = torch.max(eq_flow[t0],dim=0)
        groups[:,t0] = indices

    # -- from flow_index to a group ID --
    ngroups = 0
    for t0 in range(nframes):
        match = groups == t0
        if torch.any(match):
            groups[torch.where(match)] = ngroups
            ngroups += 1

    # -- reshape to output --
    groups = rearrange(groups,'(i h w) t -> t i h w',h=H,w=W)
    ngroups = groups.max().item()+1

    # -- for testing --
    # groups = torch.arange(nframes).to(flow.device)
    # groups = repeat(groups,'t -> t i h w',i=nimages,h=H,w=W)
    # ngroups = nframes

    return groups,ngroups

def smooth_locs(locs,nclusters=3):

    nframes,nimages,h,w,k,two = locs.shape
    locs = locs[...,0,:] # top k only 
    locs_fmt = rearrange(locs,'t i h w two -> (t i) (h w) two')
    locs_fmt = locs_fmt.contiguous().type(torch.float)

    # -- exec clustering --
    names,means,counts,dists = KMeans(locs_fmt, K=nclusters,
                                      Niter=10, verbose=False, randDist=0.)
    # -- shaping --
    means = rearrange(means,'(t i) c two -> t i c two',t=nframes,i=nimages)
    means = torch.round(means).type(torch.long)
    olocs = rearrange(torch.zeros_like(locs),'t i h w two -> t i (h w) two')
    names = rearrange(names,'(t i) hw -> t i hw',t=nframes,i=nimages)
    print(names[:,0,16*64+16])

    # -- fill in the locs with clusters --    
    olocs[...,0] = torch.gather(means[:,:,:,0],-1,names,out=olocs[...,0])
    olocs[...,1] = torch.gather(means[:,:,:,1],-1,names,out=olocs[...,1])

    # -- final shaping --
    olocs = rearrange(olocs,'t i (h w) two -> t i h w 1 two',h=h)

    return olocs

def clip_loc_boarders(locs,patchsize,l2_nblocks,nblocks):
    # -- clip boundaries --
    pad = l2_nblocks//2 + patchsize//2
    nbHalf = nblocks//2
    def lclip(boarder):
        return torch.clamp(boarder,min=-nbHalf,max=nbHalf)
    locs[:,:,:pad,:,:,:] = lclip(locs[:,:,:pad,:,:,:])
    locs[:,:,:,:pad,:,:] = lclip(locs[:,:,:,:pad,:,:])
    locs[:,:,-pad:,:,:,:] = lclip(locs[:,:,-pad:,:,:,:])
    locs[:,:,:,-pad:,:,:] = lclip(locs[:,:,:,-pad:,:,:])
    return locs

def cluster_frames_by_groups(wburst,groups,ngroups):
    # print("wburst.shape: ",wburst.shape)
    # print("groups.shape: ",groups.shape)
    nframes,nimages,F,H,W = wburst.shape
    groups = repeat(groups,'t i h w -> t i f h w',f=F)

    # print("[post] groups.shape: ",groups.shape)
    ones = torch.ones_like(wburst)
    counts = torch.zeros_like(wburst)
    counts = counts.scatter_add(0,groups,ones)
    gburst = torch.zeros_like(wburst)
    gburst = gburst.scatter_add(0,groups,wburst)/counts
    gburst = gburst[:ngroups]

    # print("groups: ")
    # print(groups[:,0,0,16,16])
    # print(counts[:,0,0,16,16])
    # print(gburst[:,0,0,16,16])
    # print(wburst[:,0,0,16,16])

    assert torch.any(torch.isnan(gburst)).item() is False, "[utils] no nan plz."
    del groups
    return gburst

def compute_mode(std,patchsize,groups):
    # counts is _per frame_ and is the same at _each pixel_

    # -- get counts --
    nframes,nimages,H,W = groups.shape
    device = groups.device
    # counts = groups[:,:,20,20]

    groups = groups[:,:,20,20]
    ones = torch.ones_like(groups)
    counts = torch.zeros_like(groups)
    ngroups = groups.max().item()+1
    counts = counts.scatter_add(0,groups,ones)
    counts = counts[:ngroups]
    assert counts.shape[0] == ngroups,"Counts is simple for now."

    # -- create modes per frame --
    index = torch.arange(ngroups)
    modes = torch.zeros(ngroups)
    var = std**2
    p = patchsize 
    for t in range(ngroups):
        not_t = torch.cat([index[slice(None,t)],index[slice(t+1,None)]],dim=0)
        Gk = counts[t]
        harm_sum = torch.sum(1/counts[not_t])/ngroups**2
        frame_term = ((ngroups-1) / ngroups)**2 * counts[t]
        c2 = var * (frame_term + harm_sum) # coeff of gamma
        mode =  (1 - 2/p) * c2 * p
        modes[t] = mode
    return modes

def grouped_flow(flow,groups):
    print("flow.shape: ",flow.shape)
    print("groups.shape: ",groups.shape)
    nimages,nframes,H,W,two = flow.shape
    groups = rearrange(groups,'t i h w -> i t h w')
    ngroups = groups.max().item()+1
    nimages,nframes,H,W = groups.shape
    groups = groups[0,:,20,20]
    gflow = torch.zeros_like(flow)[:,:ngroups]
    for g in range(ngroups):
        t = torch.min(torch.where(groups == g)[0]).item()
        gflow[:,g,:,:,:] = flow[:,t,:,:,:]
    return gflow

def compute_temporal_cluster(wburst,K):
    method = "per_frame"
    if method == "per_frame":
        return compute_temporal_cluster_per_frame(wburst,K)
    elif method == "per_pixel":
        return compute_temporal_cluster_per_pixel(wburst,K)    
    else:
        raise KeyError(f"Uknown method name [{method}]")

def compute_temporal_cluster_per_frame(wburst,K):
    # -- unpack --
    nparticles,nframes,nimages,nftrs,h,w = wburst.shape

    # -- compute per-pixel assignments --
    rburst = rearrange(wburst,'p t i f h w -> (p i) t (h w f)')
    rburst = rburst.contiguous()
    names,means,counts,dists = KMeans(rburst, K=K, Niter=10, verbose=False, randDist=0.)

    # -- shape for image sizes --
    shape_args = {'p':nparticles,'i':nimages,'h':h,'w':w}
    names = repeat(names,'(p i) t -> p t i h w',**shape_args)
    means = repeat(means,'(p i) t (h w f) -> p t i f h w',**shape_args)
    counts = repeat(counts,'(p i) t 1 -> p t i 1 h w',**shape_args)

    # -- correct for "identical" matching; a cluster might be empty --
    weights = counts/nframes
    any_empty_clusters = torch.any(counts == 0).item()
    assert any_empty_clusters == False,"No empty clusters!"
    eq_zero = counts == 0
    mask = torch.where(eq_zero,1,0).type(torch.bool)
    
    return names,means,weights,mask

def compute_temporal_cluster_per_pixel(wburst,K):
    
    # -- unpack --
    nparticles,nframes,nimages,nftrs,h,w = wburst.shape

    # -- compute per-pixel assignments --
    rburst = rearrange(wburst,'p t i f h w -> (p i h w) t f')
    rburst = rburst.contiguous()
    names,means,counts,dists = KMeans(rburst, K=K, Niter=10, verbose=False, randDist=0.)

    # -- shape for image sizes --
    shape_args = {'p':nparticles,'i':nimages,'h':h}
    names = rearrange(names,'(p i h w) t -> p t i h w',**shape_args)
    means = rearrange(means,'(p i h w) t f -> p t i f h w',**shape_args)
    counts = rearrange(counts,'(p i h w) t 1 -> p t i 1 h w',**shape_args)

    # -- correct for "identical" matching; a cluster might be empty --
    weights = counts/nframes
    any_empty_clusters = torch.any(counts == 0).item()
    assert any_empty_clusters == False,"No empty clusters!"
    eq_zero = counts == 0
    mask = torch.where(eq_zero,1,0).type(torch.bool)
    
    return names,means,weights,mask
    
def locs_frames2groups(pix,names,sranges,nblocks):
    
    # print("names.shape: ",names.shape)
    # print("pix.shape: ",pix.shape)
    # print("sranges.shape: ",sranges.shape)
    if pix.dim() == 5: pix = pix[:,None] # add batch dim if necessary

    # -- unpack shapes --
    nframes,nimages,h,w,k,two = pix.shape
    names = rearrange(names,'i t 1 h w -> t i h w 1')
    nframes,nimages,hN,wN,one = names.shape
    nsearch,h,w,nelems,two = sranges.shape
    lnames = loc_index_names(1,h,w,k,pix.device)
    locs = pix - lnames
    assert h == hN, f"[names and locs] two sizes must match: {h} vs {hN}"
    assert w == wN, f"[names and locs] two sizes must match: {w} vs {wN}"
    # nnames = loc_index_names(nframes,hN,wN,k,pix.device)
    # repeat(nnames,'1 h w k two -> i
    # print("lnames.shape ",lnames.shape)

    # nnames = repeat(torch.arange(nframes),'t -> i t 1 h w',
    #                 i=nimages,h=hN,w=wN)
    # print("nnames.shape ",nnames.shape)
    # print(nnames)

    # -- group locs by names --
    ngroups = names.max()+1
    g_locs = torch.zeros((nelems,h,w,k,two))
    SILLY_BIG = 10000
    for groupID in range(ngroups):
        # print(locs.shape,names.shape)
        x = torch.where(names == groupID,locs[...,0],SILLY_BIG)
        y = torch.where(names == groupID,locs[...,1],SILLY_BIG)
        
        # print(indices,len(indices))
        # print(torch.stack(indices).shape)
        # x = torch.where(names == groupID,locs[...,0,0],?)
        # y = torch.where(names == groupID,locs[...,0,1],?)


def slice_state_testing(locs,names,sranges,nblocks):
    
    # print("names.shape: ",names.shape)
    # print("locs.shape: ",locs.shape)
    # print("sranges.shape: ",sranges.shape)
    nframes,nimages,h,w = names.shape
    nelems,h,w,k,two = locs.shape
    nsearch,h,w,nelems,two = sranges.shape
    lnames = loc_index_names(1,h,w,k,locs.device)[0]


    blocks = locs - lnames
    x = blocks[:,32,32,0,0]
    y = blocks[:,32,32,0,1]
    sx = sranges[:,32,32,:,0]
    sy = sranges[:,32,32,:,1]
    
    nelems = x.shape
    nelems = y.shape
    nsearch,nelems = sx.shape
    nsearch,nelems = sy.shape

    x_center = sx[nsearch//2,:]
    y_center = sy[nsearch//2,:]

    nbHalf = nblocks//2
    sq_l = int(np.sqrt(nsearch))
    bG = nblocks + 2*(nblocks//2)
    grid = np.zeros((bG,bG))
    mid = bG//2
    for t in range(nelems):
        x_i,y_i = x[t],y[t]
        x_offset = int(x_i - x_center[t])
        y_offset = int(y_i - y_center[t])
        assert np.abs(x_offset) <= nbHalf,"centered must be contained."
        assert np.abs(y_offset) <= nbHalf,"centered must be contained."
        
        
        # offsets -> block index
        x_bindex = x_offset + nbHalf
        y_bindex = -y_offset + nbHalf # spatial to matrix coords

        # block index -> top-left coord
        x_tl = mid - x_bindex
        y_tl = mid - y_bindex

        # print("x,y")
        # print(x_offset,x_bindex,x_tl)
        # print(y_offset,y_bindex,y_tl)
        # ystart = max([y_i_mod-nbHalf,0])
        # yend = min([y_i_mod+nbHalf,nbHalf])
        # yslice = slice(ystart,yend)

        # xstart = max([x_i_mod-nbHalf,0])
        # xend = min([x_i_mod+nbHalf,nbHalf])
        # xslice = slice(xstart,xend)


        # ystart = y_i_mod-nbHalf+nblocks
        # yend = ystart + nblocks
        # yslice = slice(ystart,yend)

        # xstart = x_i_mod-nbHalf + nblocks
        # xend = x_i_mod+nbHalf
        # xslice = slice(xstart,xend)


        # print(yslice,xslice)
        yslice = slice(y_tl,y_tl+nblocks)
        xslice = slice(x_tl,x_tl+nblocks)
        grid[yslice,xslice] += 1

    # print(grid)
    shared = np.where(grid == nelems)
    xshared = shared[1] - nbHalf
    yshared = shared[0] - nbHalf

    # assert np.all(0 <= xshared),"all contained."
    # assert np.all(xshared < nblocks),"all contained."
    # assert np.all(0 <= yshared),"all contained."
    # assert np.all(yshared < nblocks),"all contained."

    # print(xshared)
    # print(yshared)
    


    # -- fake stuff here ... --
    # say we ran code and we get the "best" index of the group
    # best_arangement = [xshared[0],yshared[0]]
    best_arangement = [1,1]
    best_x = best_arangement[0]
    best_y = best_arangement[1]

    # print("best")
    # print(best_x,best_y)

    for t in range(nelems):
        x_i = x[t]
        y_i = y[t]
        x_offset = int(x_i - x_center[t])
        y_offset = int(y_i - y_center[t])
        assert np.abs(x_offset) <= nbHalf,"centered must be contained."
        assert np.abs(y_offset) <= nbHalf,"centered must be contained."
        
        
        # offsets -> block index
        x_bindex = x_offset + nbHalf
        y_bindex = -y_offset + nbHalf # spatial to matrix coords

        # block index -> top-left coord
        x_tl = mid - x_bindex
        y_tl = mid - y_bindex

        # OTHER WAY

        tmp_x = best_x + nbHalf
        tmp_y = best_y + nbHalf

        bl_x = tmp_x - x_tl
        bl_y = tmp_y - y_tl

        # off_x = bl_x - nbHalf
        # off_y = -bl_y + nbHalf
        # l_x = x_tl - tmp_x
        # l_y = y_tl - tmp_y

        off_x = bl_x - nbHalf
        off_y = -(bl_y - nbHalf)
        
        # print("x",x_bindex,x_tl,x_i,best_x,tmp_x,bl_x,off_x)
        # print("y",y_bindex,y_tl,y_i,best_y,tmp_y,bl_y,off_y)



    # ngroups = names.max()
    # for gid in range(ngroups):
    #     x = torch.where(names == gid,locs[...,0,0],?)
    #     y = torch.where(names == gid,locs[...,0,1],?)
    

    # new_locs = locs.clone()
    # new_locs[...,0] = torch.gather(sub_locs[...,0],0,names,out=new_locs[...,0])
    # new_locs[...,1] = torch.gather(sub_locs[...,0],0,names,out=new_locs[...,1])
    # for t in range(nframes):
    #     locs[t,...,0] = torch.where(replace,locs[t,...,0],new_locs[t,...,0])
    #     locs[t,...,1] = torch.where(replace,locs[t,...,1],new_locs[t,...,1])

    
def update_state_locs(curr,prop,names):
    """
    current locs: (nframes,..)
    prop locs: (ngroups,..)
    names: (nframes,...) with indicates for [ngroups]
    """

    # -- expand [prop] locs to [curr] locs --
    expanded = torch.zeros_like(curr)
    K = curr.shape[-2]
    for k in range(K):
        # print("prop.shape: ",prop.shape)
        # print("names.shape: ",names.shape)
        # print("curr.shape: ",curr.shape)
        # print("expanded.shape: ",expanded.shape)
        torch.gather(prop[...,k,0],0,names,out=expanded[...,k,0])
        torch.gather(prop[...,k,1],0,names,out=expanded[...,k,1])

    # -- add to previous state --
    expanded = curr + expanded

    return expanded
    
def update_state(vals,locs,prop_vals,prop_locs,names,overwrite):

    # -- unpack all shapes --
    nimages,h,w,k = vals.shape
    nframes,nimages,h,w,k,two = locs.shape

    nimages,h,w,k = prop_vals.shape
    nclusters,nimages,h,w,k,two = prop_locs.shape

    nimages,nframes,h,w = names.shape
    names = rearrange(names,'i t h w -> t i h w')
    # print("locs ",locs.shape)
    # print("prop_locs ",prop_locs.shape)
    # print("names ",names.shape)
    # print("vals.shape",vals.shape)
    # print("prop_vals.shape",prop_vals.shape)
    

    # -- index at top 1 --
    vals = vals[...,0]
    prop_vals = prop_vals[...,0]
    locs = locs[...,0,:]
    prop_locs = prop_locs[...,0,:].long()
    new_locs = torch.zeros_like(locs)
    
    # -- updating commences! --
    if overwrite: # todo: remove me.

        # -- replace values always --
        vals = prop_vals

        locs[...,0] = torch.gather(prop_locs[...,0],0,names,out=locs[...,0])
        locs[...,1] = torch.gather(prop_locs[...,1],0,names,out=locs[...,1])
        exp_locs_out = locs.clone()[...,None,:] 

    else:

        # -- replace bools --
        replace = vals > prop_vals
        # print("replace",replace[:,16,16],replace.shape)

        # -- replace when values are smaller --
        vals = torch.where(replace,prop_vals,vals)
        # print(vals[:,16,16])


        # -- expand locs from (ngroups -> nframes) using "names" --

        exp_locs = torch.zeros_like(locs)
        # print("loc.shape ",locs.shape)
        # print("names.shape ",names.shape)
        # print("prop_loc.shape ",prop_locs.shape)
        # print("exp_locs.shape ",exp_locs.shape)
        torch.gather(prop_locs[...,0],0,names,out=exp_locs[...,0])
        torch.gather(prop_locs[...,1],0,names,out=exp_locs[...,1])

        #
        # -- Update the "Delta Loc" --
        #
        
        elocs = exp_locs.clone()
        for t in range(nframes):
            elocs[t,...,0] = torch.where(replace,elocs[t,...,0],0)
            elocs[t,...,1] = torch.where(replace,elocs[t,...,1],0)

        #
        # -- New_Loc = Cur_Loc + Delta_Loc -- 
        #

        # exp_locs = locs + exp_locs

        # -- replace "locs" where val is smaller --
        for t in range(nframes):
            locs[t,...,0] = torch.where(replace,exp_locs[t,...,0],locs[t,...,0])
            locs[t,...,1] = torch.where(replace,exp_locs[t,...,1],locs[t,...,1])

    # -- append back "k" for api --
    vals = rearrange(vals,'i h w -> i h w 1')
    locs = rearrange(locs,'t i h w two -> t i h w 1 two')
    elocs = rearrange(elocs,'t i h w two -> t i h w 1 two')

    return vals,locs,elocs

        
def update_state_outliers(vals,locs,sub_vals,sub_locs,names,overwrite):

    # -- unpack all shapes --
    nimages,h,w,k = vals.shape
    nframes,nimages,h,w,k,two = locs.shape

    nimages,h,w,k = sub_vals.shape
    nsearch,nimages,h,w,k,two = sub_locs.shape

    nsearch = names.shape
    
    # -- index at top k == 1 --
    vals = vals[...,0]
    sub_vals = sub_vals[...,0]
    locs = locs[...,0,:]
    sub_locs = sub_locs[...,0,:].long()
    new_locs = torch.zeros_like(locs)

    # -- replace bools --
    # ratio = vals / sub_vals
    # coin = torch.rand(vals.shape).to(vals.device)
    # replace = coin < ratio
    replace = vals > sub_vals
    vals = torch.where(replace,sub_vals,vals)

    # -- replace with new values where necessary --
    prev_locs = locs.clone()
    for search_index_m1,search_name in enumerate(names):
        search_index = search_index_m1 + 1
        sn,si = search_name,search_index
        locs[sn,...,0] = torch.where(replace,sub_locs[si,...,0],locs[sn,...,0])
        locs[sn,...,1] = torch.where(replace,sub_locs[si,...,1],locs[sn,...,1])
    delta = torch.sum(torch.abs(prev_locs - locs)).item()

    # -- append back "k" for api --
    vals = rearrange(vals,'i h w -> i h w 1')
    locs = rearrange(locs,'t i h w two -> t i h w 1 two')

    return vals,locs


def denoise_clustered_burst(wburst,clusters,ave_denoiser):
    denoised = []
    for c in range(clusters):
        cluster = clusters[c]
        c_nframes = cluster.nframes
        c_mask =  cluster.mask
        c_denoised = ave_denoiser(wburst,c_mask,c_nframes)
        denoised.append(c_denoised)
    denoised = torch.stack(denoised,dim=0)
    return denoised

def index_along_ftrs(warped_tiled,ps,c):
    assert warped_tiled.shape[-3] == ps**2 * c, "ensure eq dims"
    shape_str = 't i (ps1 ps2 c) h w -> t i ps1 ps2 c h w'
    psHalf = ps//2
    warped_tiled = rearrange(warped_tiled,shape_str,ps1=ps,ps2=ps)
    warped = warped_tiled[:,:,psHalf,psHalf,:,:,:]
    # ps2 = ps**2
    # psMid = ps2//2
    # fIdx = torch.arange(psMid,ps2*c,ps2)
    # warped = warped_tiled[...,fIdx,:,:]
    return warped
