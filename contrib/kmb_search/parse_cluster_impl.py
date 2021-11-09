
import torch
from .utils import get_optional_field
from .centroid_update_impl import fill_sframes_ecentroids
from .kmeans_impl import sup_kmeans,run_ekmeans
from .dbscan_impl import run_dbscan
from .parse_utils import parse_ctype

def get_cluster_function(testing):
    choice = get_optional_field(testing,"cluster","fill")
    ctype = get_optional_field(testing,"cluster_centroid_type","noisy")
    offset = get_optional_field(testing,"km_offset",0.)
    km_iters = get_optional_field(testing,"km_iters",5)
    sup_km_version = get_optional_field(testing,"sup_km_version","v1")
    if choice == "fill":
        return get_fill_cycle(ctype)
    elif choice == "sup_kmeans":
        return get_sup_kmeans(ctype,sup_km_version)
    elif choice == "kmeans":
        return get_kmeans(ctype,offset,km_iters)
    elif choice == "dbscan":
        return get_dbscan(ctype,offset,km_iters)
    else:
        raise ValueError(f"Uknown [cluster] function [{choice}]")

def get_fill_cycle(ctype):

    def fill_cycle(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        output = fill_sframes_ecentroids(cimg,indices,iframes,ps)
        centroids,clusters,sizes,_,_ = output
        return centroids,clusters,sizes

    return fill_cycle

def get_sup_kmeans(ctype,sup_km_version):

    def run_sup_kmeans(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        kmeans_out = sup_kmeans(cimg,clean,indices,indices_gt,sframes,ps,sup_km_version)
        km_dists,clusters,sizes,centroids,rcl = kmeans_out
        return centroids,clusters,sizes

    return run_sup_kmeans

def get_kmeans(ctype,km_offset,km_iters):

    def wrap_run_ekmeans(noisy,clean,kmeansK,indices,inidices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        output = run_ekmeans(noisy,indices,kmeansK,ps,km_offset,niters=km_iters)
        km_dists,clusters,sizes,centroids = output
        return centroids,clusters,sizes

    return wrap_run_ekmeans

def get_dbscan(ctype,offset,km_iters):

    def wrap_run_dbscan(noisy,clean,kmeansK,indices,indices_gt,sframes,iframes,ps):
        cimg = parse_ctype(ctype,noisy,clean)
        output = run_dbscan(noisy,indices,kmeansK,ps,km_offset,niters=km_iters)
        km_dists,clusters,sizes,centroids = output
        return centroids,clusters,sizes

    return wrap_run_dbscan
