"""
Belief Propogation Search

"""

import torch
import faiss
import numpy as np
from einops import rearrange,repeat
import nnf_utils as nnf_utils
from nnf_share import padBurst
from sub_burst.run_burst import runBurstNnf

def runBpSearch(burst, patchsize, nblocks, k = 1,
                nparticles = 1, niters = 10, valMean = 0.,
                l2_nblocks = 5, l2_valMean=0.,
                blockLabels=None, ref=None,
                to_flow=False, fmt=False):

    assert nparticles == 1, "Only one particle currently supported."

    # -------------------------------
    #
    # ----    initalize fxn      ----
    #
    # -------------------------------

    def ave_denoiser(wburst,mask,nframes_per_pix):
        return torch.sum(wburst*mask,dim=0) / nframes_per_pix

    # -------------------------------
    #
    # ----    inital search      ----
    #
    # -------------------------------

    # -- 1.) run l2 local search --
    vals,locs = nnf_utils.runNnfBurst(burst, patchsize, l2_nblocks,
                                      nparticles, valMean = l2_valMean)

    # -- 2.) create local search radius from topK locs --
    nframes,h,w,k,two = locs.shape
    search_ranges = create_search_ranges(nblocks,h,w,nframes)

    # -- 3.) warp burst to top location --
    wburst = warp_burst(burst,locs)

    # -------------------------------
    #
    # --   execute random search   --
    #
    # -------------------------------

    for i in range(niters):

        # -- 1.) cluster each pixel across time --
        clusters = compute_temporal_cluster(wburst,patchsize,nblocks,i,niters)

        # -- 2.) denoise each group (e.g. use averages) --
        dclusters = denoise_clustered_burst(wburst,clusters,ave_denoiser)

        # -- 3.) select a (G_ref,G_not-ref) pair --
        tgtCluster,refCluster,tframes = select_search_pair(dclusters)

        # -- 4.) create combinatorial search blocks from search ranges  --
        search_blocks = compute_search_blocks(tgtCluster.frame_ids,
                                              refCluster.frame_ids,
                                              search_ranges)
        
        # -- 5.) exh srch over a (G_ref,G_non-ref) pair  --
        sub_vals,sub_locs = runBurstNnf(tgtCluster.burst,
                                        refCluster.denoised,
                                        tframes,
                                        in_vals=tgtCluster.vals,
                                        in_locs=tgtCluster.locs,
                                        blockLabels=search_blocks)
        # -- 6.) update vals and locs --
        vals,locs = update_state(vals,locs,sub_vals,sub_locs)

        # -- 7.) rewarp bursts --
        wburst = warp_burst(burst,locs)

    return vals,locs
