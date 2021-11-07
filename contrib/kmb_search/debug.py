
import torch

def get_optimal_search_index(indices,indices_gt,sframes):
    sindex = indices[:,sframes] == indices_gt[:,sframes,None]
    sindex = torch.all(sindex,dim=0)
    sindex = torch.all(sindex,dim=0)
    sindex = torch.where(sindex)[0]
    sindex = torch.unique(sindex)
    return sindex
