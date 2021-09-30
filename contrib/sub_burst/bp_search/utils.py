

from einops import rearrange,repeat
from nnf_share import getBlockLabelsRaw

def create_search_ranges(nblocks,h,w,nframes):

    # -- search range per pixel --
    search_ranges = getBlockLabelsRaw(nblocks) 
    search_ranges = repeat(search_ranges,'l two -> l h w t two',h=h,w=w,t=nframes)
    return search_ranges

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

def warp_burst(burst,locs):

    # -- block ranges per pixel --
    nframes,h,w,k,two = locs.shape
    nparticles = k

    # -- create offsets --
    warps = []
    for p in range(nparticles):
        warps[p] = align_from_pix(burst,locs[:,:,:,p])
    warps = torch.stack(warps)

    return warps

def compute_temporal_cluster(wburst,patchsize,
                             nblocks,i,niters):
    
    # -- unpack --
    h,w,k = vals.shape
    nframes,h,w,k,two = locs.shape
    nframes,i,c,h,w = wburst.shape
    
    # -- compute delta t's --
    delta = torch.zeros(nframes,nframes,i,h,w)
    for t0 in range(nframes):
        for t1 in range(nframes):
            wburst_i = wburst[:,i]
            dt = (wburst[t0] - wburst[t1])**2
            delta[t0,t1] = torch.mean(dt,dim=1)

    # -- compute per-pixel assignments --
    clusters_ids = torch.zeros(nframes,h,w)
    for t in range(nframes):
        delta_t = delta[t].cpu().numpy()
        np.where(delta_t)

    # -- create clusters --
    
        
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
