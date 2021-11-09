
import torch
from einops import rearrange,repeat

from .utils import get_optional_field,get_ref_centroids
from .parse_utils import get_centroids_for_ave

def get_ave_function(testing):
    choice = get_optional_field(testing,"ave","ave_centroids")
    ctype = get_optional_field(testing,"ave_centroid_type","noisy")
    if choice == "ref_centroids":
        return get_ref_centroids_fxn(ctype)
    elif choice in ["ave_centroids","ave_centroids_v1"]:
        return get_ave_centroids_v1(ctype)
    elif choice == "ave_centroids_v2":
        return get_ave_centroids_v2(ctype)
    else:
        raise ValueError(f"Uknown ave function [{choice}]")

def get_ref_centroids_fxn(ctype):
    def impl_ref_centroids(noisy,clean,_centroids,_clusters,_sizes,indices,ps):


        # -- get centroids --
        if ctype == "given":
            centroids,clusters,sizes = _centroids,_clusters,_sizes
        else:
            centroids,clusters,sizes = get_centroids_for_ave(ctype,noisy,clean,indices,ps)

        # -- aug size to modify centroids --
        t,s,h,w = clusters.shape
        c,tK,s,h,w,ps,ps = centroids.shape
        tK,s,h,w = sizes.shape

        # # -- set zero locations to nan --
        # aug_sizes = repeat(sizes,'t s h w -> c t s h w p1 p2',c=c,p1=ps,p2=ps)
        # centroids[torch.where(aug_sizes==0)]=float("nan")

        # -- create ref centroid --
        ref_centroids = get_ref_centroids(clusters,centroids,indices,ref=None)

        return ref_centroids
    return impl_ref_centroids

def get_ave_centroids_v1(ctype):
    def ave_centroids_v1(noisy,clean,centroids,clusters,sizes,indices,ps):

        # -- get centroids --
        if ctype == "given":
            centroids,clusters,sizes = _centroids,_clusters,_sizes
        else:
            centroids,clusters,sizes = get_centroids_for_ave(ctype,noisy,clean,indices,ps)

        # -- compute mean --
        ave = torch.nanmean(centroids,dim=1)

        return ave
    return ave_centroids_v1

def get_ave_centroids_v2(ctype):
    def ave_centroids_v2(noisy,clean,centroids,clusters,sizes,indices,ps):

        # -- get centroids --
        if ctype == "given":
            centroids,clusters,sizes = _centroids,_clusters,_sizes
        else:
            centroids,clusters,sizes = get_centroids_for_ave(ctype,noisy,clean,indices,ps)

        # -- compute mean --
        ave = torch.nansum(centroids*size,dim=1)/nframes

        return ave
    return ave_centroids_v2


