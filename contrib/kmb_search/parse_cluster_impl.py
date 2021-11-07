
import torch
from .utils import get_optional_field,parse_ctype

def get_cluster_function(testing):
    choice = get_optional_field(testing,"cluster","fill")
    ctype = get_optional_field(testing,"cluster_centroid_type","noisy")
    offset = get_optional_field(testing,"km_offset",0.)
    km_iters = get_optional_field(testing,"km_iters",10)
    if choice == "fill":
        return get_fill(ctype)
    elif choice == "sup_kmeans":
        return get_sup_kmeans(ctype)
    elif choice == "kmeans":
        return get_kmeans(ctype,offset,km_iters)
    else:
        raise ValueError("Uknown [cluster] function [{choice}]")

def get_fill_cycle(ctype):

    def fill_cycle(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        output = fill_sframes_ecentroids(kimg,indices,iframes,ps)
        centroids,clusters,sizes,_,_ = output
        return centroids,clusters,sizes

    return fill_cycle

def get_sup_kmeans(ctype):

    def run_sup_kmeans(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        kmeans_out = sup_kmeans(cimg,clean,indices,indices_gt,sframes,ps)
        km_dists,clusters,sizes,centroids,rcl = kmeans_out
        return centroids,clusters,sizes

    return run_sup_kmeans

def get_kmeans(ctype,km_offset,km_iters):

    def wrap_run_ekmeans(noisy,clean,kmeansK,indices,inidices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        output = run_ekmeans(noisy,indices,kmeansK,ps,km_offset,niters=km_iters)
        km_dists,clusters,sizes,centroids,rcl = kmeans_out
        return centroids,clusters,sizes

    return wrap_run_ekmeans

