
import torch
from .utils import get_optional_field


def get_ave_function(testing):
    choice = get_optional_field(testing,"ave","ave_centroids")
    if choice == "center_frame":
        return center_frame
    elif choice == "clean_center_frame":
        return clean_center_frame
    elif choice == "center_centroid":
        return center_centroid
    elif choice in ["ave_centriods","ave_centroids_v1"]:
        return ave_centroids_v1
    elif choice == "ave_centriods_v2":
        return ave_centroids_v2
    else:
        raise ValueError("Uknown ave function [{choice}]")


def center_frame(clean,noisy,centroids,clusters,sizes):
    t = noisy.shape[1]
    return noisy[:,t//2]

def clean_center_frame(clean,noisy,centroids,clusters,sizes):
    t = clean.shape[1]
    return clean[:,t//2]

def center_centroid(clean,noisy,centroids,clusters,sizes):
    t = clean.shape[1]
    return clean[:,t//2]

def ave_centroids_v1(clean,noisy,centroids,clusters,sizes):
    return torch.nanmean(centroids,dim=1)

def ave_centroids_v2(clean,noisy,centroids,clusters,sizes):
    nframes = clean.shape[1]
    return torch.nansum(centroids*size,dim=1)/nframes

