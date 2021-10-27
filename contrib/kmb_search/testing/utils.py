import torch
import numpy as np


def random_blocks(t,s,h,w):
    minHW = min(h,w)
    return torch.randint(0,minHW,(2,t,s,h,w)).type(torch.int)

def compute_gt_burst(burst,pix_hw,flow,ps,nblocks):
    psHalf = ps//2
    padOffset = nblocks//2
    patches,patches_ave = [],[]
    nframes,nimages,c,h,w = burst.shape
    for t in range(nframes):
        flow_t = flow[t]

        # startH = pix_hw[1]-flow_t[1]# - psHalf +0# dy
        startH = pix_hw[0] + flow_t[0] + padOffset # dy
        endH = startH + ps
        sliceH = slice(startH,endH)

        # startW = pix_hw[0]+flow_t[0]# - psHalf +0# dx
        startW = pix_hw[1] + flow_t[1] + padOffset # dx
        endW = startW + ps
        sliceW = slice(startW,endW)

        # print(pix_hw,startH,startW,burst.shape)
        patch_t = burst[t,0,:,sliceH,sliceW]
        patches.append(patch_t)


    patches = torch.stack(patches)
    ave = torch.sum(patches,dim=0)/nframes
    # ave = torch.zeros_like(ave)
    diff = torch.sum((patches - ave)**2/nframes).item()
    return diff

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
