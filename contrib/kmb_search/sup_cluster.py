

# -- python --
import numpy as np
from einops import rearrange,repeat
from numba import jit,njit,prange

# -- pytorch --
import torch


@njit
def sup_clusters_numba_v1(clusters,sizes,clean,indices,indices_gt,sframes,ps):

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


@njit
def sup_clusters_numba_v2(clusters,sizes,clean,indices,indices_gt,sframes,ps):

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
                        sneigh = False
                        is_tj_search = False
                        # sneigh = is_tj_search
                        # for tk in range(t):
                        #     if cids[ti] != cids[tk]: continue
                        #     sneigh = sbool[tk] or sneigh
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

