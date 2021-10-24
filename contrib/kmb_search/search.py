
"""
Execute search using KMeans Burst

"""


# -- python/pytorch --
import torch
from einops import rearrange,repeat

# -- pairwise l2 search imports --
from nnf_utils import runNnfBurst as runPairBurst

# -- run kmburst --
from .impl import runKmBurstSearch
from .utils import jitter_traj_ranges,init_zero_locs

def create_trajectories(locs):
    """
    locs -> trajectories

    These are approximate 
    paths of the pixel's patch 
    across time

    shaped:

    J x B x T x H x W x 2

    J = # of trajectories
    B = batchsize
    T = # of frames
    H = height
    W = width
    2 = [y,x] direction ?(or [x,y])...
    
    """
    trajs = rearrange(locs,'t i h w k two -> k i two t h w')        
    return trajs

def choose_best(agg_vals,agg_locs):
    """
    Choose the best.

    """
    return agg_vals[0],agg_locs[0]

def runKmSearch(burst,patchsize,nblocks,k=1,std=None,
               l2_patchsize=None,l2_nblocks=None,l2_k=None,
               search_space=None,ref=None):

    # -- init vars --
    device = burst.device
    nframes,nimages,c,h,w = burst.shape
    if ref is None: ref = nframes//2
    if l2_patchsize is None: l2_patchsize = patchsize
    if l2_nblocks is None: l2_nblocks = nblocks
    if l2_k is None: l2_k = k

    # -- run l2 --
    # l2_mode = compute_l2_mode(std,patchsize)
    # vals,locs = runPairBurst(burst,l2_patchsize,
    #                          l2_nblocks,k=l2_k,valMean=l2_mode,
    #                          img_shape = None)
    locs = init_zero_locs(nframes,nimages,h,w).to(device)

    # -- create a set of smoothed trajectories --
    trajs = create_trajectories(locs)
    search_ranges = jitter_traj_ranges(trajs,nblocks)
    print(locs.shape)
    print(search_ranges.shape)
    print(search_ranges[...,16,16])

    # -- create search space from trajectories --
    nframes_search = 3 # constant from C++
    nsearch = nblocks**2#**(nframes_search)
    agg_vals,agg_locs = [],[]
    for search_ranges_p in search_ranges:
        vals_p,locs_p = runKmBurstSearch(burst, patchsize, nsearch,
                                         k=1, kmeansK=3, ref = ref,
                                         search_ranges = search_ranges_p)
        agg_vals.append(vals_p)
        agg_locs.append(locs_p)
    
    # -- choose the best from the trajectories --
    vals,locs = choose_best(agg_vals,agg_locs)

    return vals,locs


"""


-> Trajectory or _Search Ranges_ or Meshgrid?
-> Meshgrid:
   -> we can't do a meshgrid since we don't know the clusters in advance.
   -> any meshgrid using all the frames is too big
   -> this is out.
-> Trajectories:
   -> only requies a pair of numbers per HxWxT
   -> uses implicit search range
-> Search Ranges
   -> allows for trajectories to be used
   -> allows for not just local searches to be used


--------------
--> Steps: <--
--------------

   0.) fix frames at a state and search over others.
      -> requires some concept of state
      -> ....?

   1.) batch across (a) H, (b) W, and (c) Blocks
      -> what does it mean to get a batch of "blocks" if the blocks are not 
         already meshed together?
      -> it seems like we need to know the "number" of blocks and then we
         can somehow create the correct blocks for the given (start,end) integers?
      batched_search_mesh = function(search_ranges,start,end,...)

   2.) compute the clusters for the given proposed alignments
      -> for each proposed search, we compute need the 
         (i) centroid frames 
         (ii) [NOT NEEDED] --cluster--assignments--; no use in this case

   3.) we compute the final quality: \ave_i((c_i - c_bar)) 

   4.) add to topK


-> WAIT: the optimal should only have 1 cluster with all the frames...
-> However, our measure of goodness is an inter-cluster distance measure
-> So we by computing the "centroid" to the "global centroid mean", we
   create a "hedged" estimate of the final result.
-> Think if K = T, then we reduce the sample variance
-> Think if K = 1, then we reduce to 0 identity
-> This is kind of like the resolution? 
   -> If we knew the clusters, we get the best possible resolution?
   -> If we get incorrect clusters, our result is slightly wrong.
      -> If all but one frame in cluster is aligned, then inaccuracy has small impact
      -> If two modes are equally represented, we create a new mode 
-> So if 1 < K < T, then we balance between finding multiple modes? and the single mode?


"""
