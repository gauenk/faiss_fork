
# -- python --
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- pytorch --
import torch

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#              Supervised Clustering
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def sup_clusters(clean,indices,indices_gt,sframes,ps):
    
    # -- init vars --
    device = indices.device
    two,t,s,h,w = indices.shape
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((t,s,h,w)).astype(np.uint8)

    # -- launch numba --
    clean = clean.cpu().numpy()
    indices = indices.cpu().numpy()
    indices_gt = indices_gt.cpu().numpy()
    sframes = sframes.cpu().numpy()
    # print("pre numba: ",sframes)
    sup_clusters_numba(clusters,sizes,clean,indices,indices_gt,sframes,ps)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def sup_clusters_numba(clusters,sizes,clean,indices,indices_gt,sframes,ps):

    def bounds(p,lim):
        if p < 0: p = -p
        if p > lim: p = 2*lim - p
        # p = p % lim
        return p

    # -- shape --
    nsframes = len(sframes)
    f,t,h,w = clean.shape
    two,t,s,h,w = indices.shape    
    psHalf,ref = ps//2,t//2
    

    for hi in prange(h):
        for wi in prange(w):
            for si in prange(s):
                cids,count = -np.ones(t,dtype=np.int32),0
                sbool = np.zeros(t,dtype=np.int32)
                for ti in range(t):

                    # -- some bools --
                    is_ti_search = sbool[ti] == 1
                    for sti in range(nsframes):
                        if ti == sframes[sti]: is_ti_search = True
                    
                    # -- create new cluster if not already assigned --
                    if cids[ti] == -1: #sbool[ti] should be false
                        cids[ti] = count
                        sbool[ti] = sbool[ti] or is_ti_search
                        count += 1
                    else: continue

                    for tj in range(ti+1,t):

                        #
                        # -- are these aligned right now ? --
                        #

                        # -- get indices --
                        i0 = indices[0,ti,si,hi,wi]
                        i1 = indices[1,ti,si,hi,wi]
                        i_top,i_left = i0-psHalf,i1-psHalf

                        j0 = indices[0,tj,si,hi,wi]
                        j1 = indices[1,tj,si,hi,wi]
                        j_top,j_left = j0-psHalf,j1-psHalf

                        # -- compare over blocks --
                        aligned = True
                        for fi in range(f):
                            for pi in range(ps):
                                for pj in range(ps):
                                    iH = bounds(i_top+pi,h-1)
                                    iW = bounds(i_left+pj,w-1)
                                    jH = bounds(j_top+pi,h-1)
                                    jW = bounds(j_left+pj,w-1)
                                    ic = clean[fi,ti,iH,iW]
                                    jc = clean[fi,tj,jH,jW]
                                    aligned = (ic == jc) and aligned

                        #
                        # -- label if aligned --
                        #

                        if not(aligned): continue # skip

                        # -- some bools --
                        is_tj_search = sbool[tj] == 1
                        for stj in range(nsframes):
                            if tj == sframes[stj]: is_tj_search = True

                        # if hi == 1 and wi == 5 and si == 77:
                        #     print(si,ti,tj,is_tj_search)
                        # if hi == 1 and wi == 5 and si == 21:
                        #     print(si,ti,tj,is_tj_search)

                        # -- are any "search" neighbors? --
                        sneigh = is_tj_search
                        for tk in range(t):
                            if cids[ti] != cids[tk]: continue
                            sneigh = sbool[tk] or sneigh
                        # msg = "must be eq.\n\n\n"
                        # msg_a = "sneigh,is_tj_search: (%d,%d) "%(sneigh,is_tj_search)
                        # if is_tj_search != sneigh:
                        #     print(sneigh,is_tj_search)
                        #     print(cids)
                        #     print(ti,tj)
                        #     print(cids[ti])
                        #     print(sbool)
                        #     # print(msg_a)
                        # msg = "help! I am trapped in a computer."
                        # assert is_tj_search == sneigh,msg
                            
                        # -- don't cluster two search frames --
                        if sneigh and is_tj_search:
                            continue
                        # if sneigh and sbool[ti] == 1:
                        #     continue
    
                        # -- share the current label --
                        if cids[tj] == -1:
                            cids[tj] = cids[ti]
                            for tk in range(t):
                                if cids[tk] != cids[ti]: continue
                                sbool[tk] = is_tj_search or sneigh#is_tj_search
                        elif cids[tj] == cids[ti]:
                            pass
                        elif sbool[tj] == sbool[ti]:
                            pass
                        else:
                            # print("huh?")
                            pass

                # if hi == 1 and wi == 5 and si == 77:
                #     print(hi,wi,si)
                #     print(cids)
                #     print(sbool)
                #     print(sframes)
                # if hi == 1 and wi == 5 and si == 21:
                #     print(hi,wi,si)
                #     print(cids)
                #     print(sbool)
                #     print(sframes)

                # -- assign clusters --
                for ti in range(t):
                    clusters[ti][si][hi][wi] = cids[ti]
                    sizes[cids[ti]][si][hi][wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#              Cluster Update using Pairwise Distances
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def update_clusters(dists):

    # -- unpack --
    device = dists.device
    t,tK,s,h,w = dists.shape
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    dists = dists.cpu().numpy()
    update_clusters_numba(dists,clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def update_clusters_numba(dists,clusters,sizes):
    t,tK,s,h,w = dists.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t0_argmin = -1
                    for t1 in range(tK):
                        d = dists[t0,t1,si,hi,wi]
                        if d < dmin:
                            dmin = d
                            t0_argmin = t1
                    clusters[t0,si,hi,wi] = t0_argmin
                    sizes[t0_argmin,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Init Cluster Update
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def init_clusters(t,tK,s,h,w,device='cuda:0'):

    # -- unpack --
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    init_clusters_numba(clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def init_clusters_numba(clusters,sizes):
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t1 = t0 % tK
                    clusters[t0,si,hi,wi] = t1
                    sizes[t1,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Randomly Init Clusters
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def rand_clusters(t,tK,s,h,w,device='cuda:0'):

    # -- unpack --
    clusters = np.zeros((t,s,h,w)).astype(np.uint8)
    sizes = np.zeros((tK,s,h,w)).astype(np.uint8)
    
    # -- numba --
    rand_clusters_numba(clusters,sizes)

    # -- to torch --
    clusters = torch.ByteTensor(clusters).to(device)
    sizes = torch.ByteTensor(sizes).to(device)

    return clusters,sizes

@njit
def rand_clusters_numba(clusters,sizes):
    t,s,h,w = clusters.shape
    tK = sizes.shape[0]
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t1 = int(np.random.rand(1)[0]*tK)
                    clusters[t0,si,hi,wi] = t1
                    sizes[t1,si,hi,wi] += 1

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
#                Self Similar Cluster Update
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def update_self_clusters(dists):

    # dists from self_pairwise_distance

    # -- unpack --
    device = dists.device
    t,t,s,h,w = dists.shape
    clusters = np.zeros((t,s,h,w)).astype(np.int)
    
    # -- numba --
    dists = dists.cpu().numpy()
    update_self_clusters_numba(dists,clusters)

    # -- to torch --
    clusters = torch.IntTensor(clusters).to(device)

    return clusters

@njit
def update_self_clusters_numba(dists,clusters):

    def reflect_pair(t0,t1):
        if t1 > t0:
            r0 = t1
            r1 = t0
        else:
            r0 = t0
            r1 = t1
        return r0,r1

    t,t,s,h,w = dists.shape
    for si in prange(s):
        for hi in prange(h):
            for wi in prange(w):
                for t0 in prange(t):
                    dmin = np.inf
                    t0_argmin = -1
                    for t1 in range(t):
                        if t0 == t1: continue
                        r0,r1 = reflect_pair(t0,t1)
                        d = dists[r0,r1,si,hi,wi]
                        if d < dmin:
                            dmin = d
                            t0_argmin = t1
                    clusters[t0,si,hi,wi] = t0_argmin

