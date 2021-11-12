

# -- python --
from einops import rearrange,repeat

# -- kmb search imports --
from kmb_search.centroid_update_impl import update_ecentroids
from kmb_search.cluster_update_impl import init_clusters

def return_optional_params(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default
    
def select_search_image(noisy,gt_info,params):
    clean = return_optional_params(gt_info,"clean",None)
    itype = return_optional_params(params,"nnf_itype","noisy")
    if clean is None or itype == "noisy": return noisy
    elif not(clean is None) and itype == "clean": return clean
    else: raise ValueError(f"Uknown itype [{itype}]")

def expanded_to_burst(eburst,h,w,ps):
    psHalf = ps//2
    eburst = rearrange(eburst,'(h w) (c p1 p2) t -> c t h w p1 p2',h=h,w=w,p1=ps,p2=ps)
    burst = eburst[...,psHalf,psHalf]
    return burst

def expand_burst(burst,inds,ps):
    device = burst.device
    c,t,h,w = burst.shape
    _clusters,_sizes = init_clusters(t,t,1,h,w,device)
    eburst = update_ecentroids(burst,inds,_clusters,_sizes,ps)
    eburst = rearrange(eburst,'c t s h w p1 p2 -> (h w) (c p1 p2) (t s)')
    return eburst
