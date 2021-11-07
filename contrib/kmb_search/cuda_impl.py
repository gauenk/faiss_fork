
# -- python --
import tqdm,math,time
import numpy as np
from einops import rearrange,repeat

# -- project --
from pyutils import save_image,get_img_coords

# -- pytorch --
import torch
import torchvision.transforms.functional as tvF

# -- faiss --
import faiss
from warp_utils import locs2flow,flow2locs,pix2locs,pix2flow

# -- kmb functionality --
from .kmeans_impl import run_ekmeans
# from .cuda.kmeans_impl import run_kmeans
from .cuda.l2norm_impl import compute_l2norm_cuda
from .cuda.subave_impl import compute_subset_ave
from .cuda.weights_impl import compute_weights
from .topk_impl import kmb_topk_update,update_state,topk_torch
from .utils import jitter_search_ranges,tiled_search_frames,mesh_from_ranges,divUp
from .utils import initialize_indices,pick_fill_frames,get_gt_info


def wait_streams(waiting,waitOn):

    # -- create events for each stream --
    events = []
    for stream in waitOn:
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        # stream.record_event(event)
        events.append(event)

    # -- issue wait for all streams in waitOn and all events --
    for stream in waiting:
        for event in events:
            # stream.wait_event(event)
            event.wait(stream)

def get_hw_batches(h,w,bsize):
    hbatches = torch.arange(0,h,bsize)
    wbatches = torch.arange(0,w,bsize)
    return hbatches,wbatches

def view_batch(tensor,h_start,w_start,size):
    hslice = slice(h_start,h_start+size)
    wslice = slice(w_start,w_start+size)
    return tensor[...,hslice,wslice]

def vprint(*args,**kwargs):
    VERBOSE = False
    if VERBOSE: print(*args,**kwargs)

def run_kmb_cuda(res, burst , patchsize, nsearch, k,
                 kmeansK, std, ref, search_ranges,
                 nsearch_xy=3, nsiters=2, nfsearch=4,
                 gt_info=None,testing=None):
    
    # -- defaults --
    t,c,h,w = burst.shape
    if nsearch_xy is None: nsearch_xy = 3
    if nfsearch is None: nfsearch = 5
    if nsiters is None: nsiters = divUp(t,nfsearch)
    if not(gt_info is None): indices_gt = gt_info['indices']
    else: indices_gt = None
    if search_ranges is None:
        search_ranges = jitter_search_ranges(nsearch_xy,t,h,w).to(device)
    nfsearch = 4
    nsiters = divUp(t,nfsearch)
    nframes = t
    alpha = 2

    # -- shape --
    device = burst.device
    t,c,h,w = burst.shape
    kmeansK = 3
    burst = rearrange(burst,'t c h w -> c t h w')
    ps = patchsize
    Z_l2 = ps*ps*c
    coords = get_img_coords(t,1,h,w)[:,:,0].to(device)
    
    # -- outputs --
    outT = nfsearch
    vals = torch.ones(outT,k,h,w).to(device)*float("nan")#1000
    inds = coords[:,:,None].repeat(1,1,k,1,1).type(torch.long) # 2,t,k,h,w
    modes = torch.zeros(outT,k,h,w).to(device)
    cmodes = torch.zeros_like(vals)

    # -- search frames --
    search_frames = tiled_search_frames(t,nfsearch,nsiters,ref).to(device)
    search_frames = search_frames.type(torch.long)

    # -- set current -- 
    curr_indices = initialize_indices(coords,search_ranges,indices_gt)

    # -- batching height and width --
    bsize = 16
    hbatches,wbatches = get_hw_batches(h,w,bsize)

    # -- search subsets "nisters" times --
    indices_match = []
    kSched = np.array([kmeansK,]*nsiters)
    assert len(kSched) >= nsiters,"k sched and nisters."
    assert np.all(kSched <= t),"all less than num frames."
    prev_sframes = search_frames[0].clone()

    # -- create streams --
    torch.cuda.synchronize()
    curr_stream = 0
    nstreams = 1
    streams = [torch.cuda.default_stream()]
    # streams = [torch.cuda.Stream()]
    streams += [torch.cuda.Stream(device=device,priority=0) for s in range(nstreams-1)]
    wait_streams(streams,[streams[curr_stream]])
    # torch.cuda.synchronize()
    # torch.cuda.set_stream(streams[curr_stream])
    if nstreams > 0:
        for s in streams: s.synchronize()
    # print(streams)

    # -- stream buffers --
    l2_bufs = [None,]*nstreams
    ave_bufs = [None,]*nstreams
    inds_bufs = [None,]*nstreams
    burstView_bufs = [None,]*nstreams
    indexView_bufs = [None,]*nstreams
    cindexView_bufs = [None,]*nstreams
    
    for s_iter in tqdm.tqdm(range(nsiters)):#nsiters

        #-- set stream info --
        curr_stream = 0
        torch.cuda.set_stream(streams[curr_stream])

        # -- prints --
        # vprint(f"Iteration: {s}")
        # vprint(search_frames[s])

        # -- iter info --
        kmeansK = kSched[s_iter]
        sframes = search_frames[s_iter]
            
        # -- meshgrid --
        indices = mesh_from_ranges(search_ranges,sframes,curr_indices,ref)
        indices = indices.to(device).type(torch.long)

        # -- fill subburst with vals --
        fgrid = pick_fill_frames(sframes,nfsearch,nframes,alpha,s_iter,device)
        # fgrid = sframes
        weights = compute_weights(fgrid,nframes)

        # -- batch across image res in different streams --
        filled = torch.zeros(h,w)

        # -- synch before start --
        torch.cuda.synchronize()

        for h_start in hbatches:
            for w_start in wbatches:

                # -- fill for checking --
                # fill_b = view_batch(filled,h_start,w_start,bsize)
                # fill_b[...] += 1.

                # -- assign to stream --
                # print(streams[curr_stream])
                torch.cuda.synchronize(0)
                cs = curr_stream 
                # print(cs)
                torch.cuda.current_stream().synchronize()
                # streams[curr_stream].wait(streams[curr_stream])
                torch.cuda.set_stream(streams[curr_stream])
                torch.cuda.synchronize(0)

                # -- grab data of batch --
                # sranges_b = view_batch(search_ranges,h_start,w_start,bsize)
                # burst_b = view_batch(burst,h_start,w_start,bsize)
                # indices_b = view_batch(indices,h_start,w_start,bsize)
                # cindices_b = view_batch(curr_indices,h_start,w_start,bsize)
                
                # -- grab data of batch --
                burstView_bufs[cs] = view_batch(burst,h_start,w_start,bsize)
                indexView_bufs[cs] = view_batch(indices,h_start,w_start,bsize)
                cindexView_bufs[cs] = view_batch(curr_indices,h_start,w_start,bsize)

                # -- kmeans --
                # pwd_mode = 0.
                # weights_bufs[cs] = run_ekmeans(burst,indexView_bufs[cs],
                #                                kmeansK,ps,pwd_mode,niters=5)

                # -- ave using weights --
                # a_weights = weights.clone()
                # a_weights[...] = 0
                # a_weights[ref] = 1.
                ave_bufs[cs] = compute_subset_ave(burst,indexView_bufs[cs],weights,ps)
                # print(burst_b.shape)
                # print(indices_b.shape)
                # ave = torch.zeros(t,ps,ps,indices_b.shape[2],bsize,bsize)

                # -- [testing only] --
                # _,_,ave_test = fill_sframes_ecentroids(burst,indices_b,fgrid,ps)
                # ave_test = rearrange(ave_test,'t s h w p1 p2 -> t p1 p2 s h w')
                
                # -- compute modes --
                # cmodes,_ = compute_mode(std,c,ps,nframes,type='burst')
        
                # -- compute difference --
                # streams[curr_stream].synchronize()
                l2_bufs[cs] = compute_l2norm_cuda(burst,indexView_bufs[cs],
                                                  weights,ave_bufs[cs])
                # streams[curr_stream].synchronize()
                # l2_vals_i = torch.rand(indices_b.shape[2],bsize,bsize).to(device)
                l2_bufs[cs] /= Z_l2
        
                # print(weights)
                # print(burst_b[0,:,3:5,4])
                # print("ave.shape: ",ave.shape)
                # print(ave[0,:,:,0,4,4])
                # print("ave_test.shape: ",ave_test.shape)
                # print(ave_test[0,:,:,0,4,4])
                # dave = torch.mean(torch.abs(ave_test - ave)).item()
                # print("Delta Aves: ",dave)
                # print("l2_vals.shape: ",l2_vals.shape)
                # print("(h_start,w_start): (%d,%d)" % (h_start,w_start))
                # print(l2_vals[0,:,:])
                # if w_start > 0: exit()

                # -- create top k --
                # streams[curr_stream].synchronize()
                _,_,inds_bufs[cs] = topk_torch(l2_bufs[cs],vals,cmodes,
                                               indexView_bufs[cs],1)
                # inds = indices_b[:,:,[-1]]

                # -- update current state --
                # if curr_stream == 0:
                #     print("-- pre --")
                #     print(cindices_b.shape)
                #     print(cindices_b)

                cindexView_bufs[cs][...] = inds_bufs[cs][:,:,0]
                
                # if curr_stream == 0:
                #     print("-- post --")
                #     print(cindices_b)

                # torch.cuda.current_stream().wait_stream(streams[curr_stream])

                # -- change stream --
                if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

        # -- wait for all streams --
        wait_streams([streams[curr_stream]],streams)
        # torch.cuda.synchronize()
        # for s in streams:
        #     s.synchronize()#torch.cuda.current_stream().wait_stream(s)

    # -- check if image tiling makes sense --
    # print("Visited all points? ",torch.all(filled==1.).item())

    # -- print indices of matching methods --
    # if not(indices_gt is None):
    #     indices_match = torch.all((curr_indices == indices_gt),dim=0)
    #     indices_match = indices_match.type(torch.float)
    #     indices_match = indices_match[None,:,None] # add dims for save
    #     # indices_match = torch.stack(indices_match,dim=0)[:,:,None]
    #     for t in range(indices_match.shape[1]):
    #         save_image(f"tkmb_indices_{t}.png",indices_match[:,t])

    # -- swap dim --
    def swap_2d_dim(tensor,dim):
        tensor = tensor.clone()
        tmp = tensor[0].clone()
        tensor[0] = tensor[1].clone()
        tensor[1] = tmp
        return tensor

    # -- convert to pix --
    inds = curr_indices[:,:,None]
    inds = swap_2d_dim(inds,0)
    inds = rearrange(inds,'two t k h w -> t 1 h w k two')
    locs = inds
    # locs = pix2locs(inds)
    locs = pix2flow(inds)
    locs = rearrange(locs,'t 1 h w k two -> two t k h w')

    return vals,locs,modes
