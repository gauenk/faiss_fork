

import torch
import numpy as np
from einops import rearrange,repeat
from nnf_share import getBlockLabelsRaw,loc_index_names
from align.xforms import align_from_pix
from .th_kmeans import KMeans
from .meshgrid_per_pixel import meshgrid_per_pixel_launcher


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

def create_search_ranges(nblocks,h,w,nframes):

    # -- search range per pixel --
    search_ranges = getBlockLabelsRaw(nblocks) 
    search_ranges = repeat(search_ranges,'l two -> l h w t two',h=h,w=w,t=nframes)
    search_ranges[...,nframes//2,:] = 0
    return search_ranges

def compute_search_blocks(sranges,refG):

    # -- init shapes and smesh -- 
    print(sranges.shape)
    nsearch_per_group,nimages,h,w,ngroups,two = sranges.shape
    nsearch = nsearch_per_group**(ngroups-1) # since ref only has 1 elem
    smesh = torch.zeros(nsearch,nimages,h,w,ngroups,two).to(sranges.device)

    # -- numba-fy values --
    for i in range(nimages):
        meshgrid_per_pixel_launcher(sranges[:,i],smesh[:,i],refG)

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

def warp_burst(burst,locs,nblocks):

    # -- block ranges per pixel --
    nframes,nimages,h,w,k,two = locs.shape
    nparticles = k
    pix = rearrange(locs,'t i h w p two -> p i (h w) t two')

    # -- create offsets --
    warps = []
    for p in range(nparticles):
        warped = align_from_pix(burst,pix[p],nblocks)
        warps.append(warped)
    warps = torch.stack(warps).to(burst.device)

    return warps

def compute_temporal_cluster(wburst,K):
    
    # -- unpack --
    nparticles,nframes,nimages,nftrs,h,w = wburst.shape

    # -- compute per-pixel assignments --
    rburst = rearrange(wburst,'p t i f h w -> (p i h w) t f')
    rburst = rburst.contiguous()
    names,means,counts = KMeans(rburst, K=K, Niter=10, verbose=False)

    # -- shape for image sizes --
    shape_args = {'p':nparticles,'i':nimages,'h':h}
    # print("names",names.shape)
    # print("means",means.shape)
    # print("counts",counts.shape)
    # batched_scatter_index(names, rburst, nftrs)
    names = rearrange(names,'(p i h w) t -> p t i h w',**shape_args)
    means = rearrange(means,'(p i h w) t f -> p t i f h w',**shape_args)
    counts = rearrange(counts,'(p i h w) t 1 -> p t i 1 h w',**shape_args)

    # -- correct for "identical" matching; a cluster might be empty --
    weights = counts#/nframes
    # print(counts[...,32,32])
    # print(counts[...,32,32])
    # print("Any empty clusters? ",torch.any(counts == 0).item())
    # print(means[...,:2,32,32])
    # print(wburst[...,:2,32,32])
    any_empty_clusters = torch.any(counts == 0).item()
    assert any_empty_clusters == False,"No empty clusters!"
    eq_zero = counts == 0
    mask = torch.where(eq_zero,1,0).type(torch.bool)
    
    return names,means,weights,mask
    
def locs_frames2groups(pix,names,sranges,nblocks):
    
    print("names.shape: ",names.shape)
    print("pix.shape: ",pix.shape)
    print("sranges.shape: ",sranges.shape)
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
        print(locs.shape,names.shape)
        x = torch.where(names == groupID,locs[...,0],SILLY_BIG)
        y = torch.where(names == groupID,locs[...,1],SILLY_BIG)
        
        # print(indices,len(indices))
        # print(torch.stack(indices).shape)
        # x = torch.where(names == groupID,locs[...,0,0],?)
        # y = torch.where(names == groupID,locs[...,0,1],?)


def slice_state_testing(locs,names,sranges,nblocks):
    
    print("names.shape: ",names.shape)
    print("locs.shape: ",locs.shape)
    print("sranges.shape: ",sranges.shape)
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

    print(grid)
    shared = np.where(grid == nelems)
    xshared = shared[1] - nbHalf
    yshared = shared[0] - nbHalf

    # assert np.all(0 <= xshared),"all contained."
    # assert np.all(xshared < nblocks),"all contained."
    # assert np.all(0 <= yshared),"all contained."
    # assert np.all(yshared < nblocks),"all contained."

    print(xshared)
    print(yshared)
    


    # -- fake stuff here ... --
    # say we ran code and we get the "best" index of the group
    # best_arangement = [xshared[0],yshared[0]]
    best_arangement = [1,1]
    best_x = best_arangement[0]
    best_y = best_arangement[1]

    print("best")
    print(best_x,best_y)

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
        
        print("x",x_bindex,x_tl,x_i,best_x,tmp_x,bl_x,off_x)
        print("y",y_bindex,y_tl,y_i,best_y,tmp_y,bl_y,off_y)



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

    

def update_state(vals,locs,sub_vals,sub_locs,names,overwrite):

    # -- unpack all shapes --
    nimages,h,w,k = vals.shape
    nframes,nimages,h,w,k,two = locs.shape

    nimages,h,w,k = sub_vals.shape
    nclusters,nimages,h,w,k,two = sub_locs.shape

    nframes,nimages,h,w = names.shape
    # names = rearrange(names,'i t h w -> t i h w')
    # print("locs ",locs.shape)
    # print("sub_locs ",sub_locs.shape)
    # print("names ",names.shape)
    # print("vals.shape",vals.shape)
    # print("sub_vals.shape",sub_vals.shape)
    

    # -- index at top 1 --
    vals = vals[...,0]
    sub_vals = sub_vals[...,0]
    locs = locs[...,0,:]
    sub_locs = sub_locs[...,0,:].long()
    

    # -- updating commences! --
    if overwrite:

        # -- replace values always --
        vals = sub_vals
        locs[...,0] = torch.gather(sub_locs[...,0],0,names,out=locs[...,0])
        locs[...,1] = torch.gather(sub_locs[...,1],0,names,out=locs[...,1])

    else:

        # -- replace bools --
        replace = vals < sub_vals

        # -- replace when values are smaller --
        vals = torch.where(replace,vals,sub_vals)

        # -- replace all locs into tmp: only update when val says
        new_locs = locs.clone()
        new_locs[...,0] = torch.gather(sub_locs[...,0],0,names,out=new_locs[...,0])
        new_locs[...,1] = torch.gather(sub_locs[...,0],0,names,out=new_locs[...,1])
        for t in range(nframes):
            locs[t,...,0] = torch.where(replace,locs[t,...,0],new_locs[t,...,0])
            locs[t,...,1] = torch.where(replace,locs[t,...,1],new_locs[t,...,1])

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
