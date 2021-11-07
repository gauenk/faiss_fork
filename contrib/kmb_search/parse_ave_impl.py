
import torch

from .utils import get_optional_field,parse_ctype
from .cluster_update_impl import init_clusters
from .centroid_update_impl import fill_sframes_ecentroids
from .centroid_update_impl import update_ecentroids

def get_ave_function(testing):
    choice = get_optional_field(testing,"ave","ave_centroids")
    ctype = get_optional_field(testing,"ave_centroid_type","noisy")
    if choice == "ref_centroid":
        return get_ref_centroid(ctype)
    elif choice in ["ave_centriods","ave_centroids_v1"]:
        return get_ave_centroids_v1(ctype)
    elif choice == "ave_centriods_v2":
        return get_ave_centroids_v2(ctype)
    else:
        raise ValueError("Uknown ave function [{choice}]")


def get_ref_centroid(ctype):
    def ref_centroid(noisy,clean,centroids,clusters,sizes,indices,ps):
        device = noisy.device
        c,nframes,h,w = burst.shape
        nsearch = indices.shape[2]
        pimg = parse_ctype(ctype,noisy,clean)
        clusters,sizes = init_clusters(nframes,nframes,nsearch,h,w,device)
        centroids = update_ecentroids(pimg,indices,clusters,tsizes,ps)
        ave = centroids[:,ref]
        return ave
    return ref_centroid
     
def get_ave_centroids_v1(ctype):
    def ave_centroids_v1(noisy,clean,centroids,clusters,sizes,indices,ps):
        return torch.nanmean(centroids,dim=1)
    return ave_centroids_v1

def get_ave_centroids_v2(ctype):
    def ave_centroids_v2(noisy,clean,centroids,clusters,sizes,indices,ps):
        nframes = clean.shape[1]
        return torch.nansum(centroids*size,dim=1)/nframes
    return ave_centroids_v2


