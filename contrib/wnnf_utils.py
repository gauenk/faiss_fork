
def runWeightedBurstNnf(burst, weights, patchsize, nblocks, k = 1,
                        valMean = 0., blockLabels=None, ref=None,
                        to_flow=False, fmt=False, in_vals=None,in_locs=None,
                        tile_burst=False):

    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    img_shape = (c,h,w)
    device = burst.device
    if ref is None: ref = nframes//2

    # # -- get block labels once for the burst if "None" --
    # blockLabels,_ = getBlockLabels(blockLabels,nblocks,torch.long,
    #                                device,True,nframes)

    # -- compute search "blockLabels" across image burst --
    vals,locs = [],[]
    for i in range(nimages):

        # -- create padded burst --
        weights_i = weights[:,i] # todo: pad weights
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)
        if tile_burst:
            burstPad_i = tileBurst(burstPad_i,h,w,patchsize,nblocks)
            img_shape = list(img_shape)
            img_shape[0] = burstPad_i.shape[1]
            input_ps = 1
        else:
            input_ps = patchsize

        # -- assign input vals and locs --
        vals_i,locs_i = in_vals,in_locs
        if not(in_vals is None): vals_i = vals_i[i]
        if not(in_locs is None): locs_i = locs_i[i]

        # -- execute over search space! --
        vals_i,locs_i = _runBurstNnf(res, img_shape, burstPad_i,
                                     weights_i,
                                     ref, vals_i, locs_i,
                                     input_ps, nblocks,
                                     k = k, valMean = valMean,
                                     blockLabels = blockLabels)

        vals.append(vals_i)
        locs.append(locs_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=0)
    # (nimages, nframes, h, w, k, two)

    if to_flow:
        locs_y = locs[...,0]
        locs_x = locs[...,1]
        locs = torch.stack([locs_x,-locs_y],dim=-1)
    
    if fmt:
        vals = rearrange(vals,'i h w k -> i (h w) k').cpu()
        locs = rearrange(locs,'i t h w k two -> k i (h w) t two').cpu().long()

    return vals,locs

def _runWeightedBurstNnf(res, img_shape, burst, weights, ref, vals, locs, patchsize, nblocks, k = 3, valMean = 0., blockLabels=None):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    ref : array_like
        Burst of images, shape (t, c, h, w).
        `dtype` must be float32.
    k : int
        Number of nearest neighbors.
    patchsize : int
        Size of patch in a single direction, a total of patchsize**2 pixels
    nblocks : int
        Number of neighboring patches to search in each direction, a total of nblocks**2
    D : array_like, optional
        Output array for distances of the nearest neighbors, shape (height, width, k)
    I : array_like, optional
        Output array for the nearest neighbors field, shape (height, width, k, 2).
        The "flow" goes _from_ refernce _to_ the target image.

    Returns
    -------
    vals : array_like
        Distances of the nearest neighbors, shape (height, width, k)
    locs : array_like
        Labels of the nearest neighbors, shape (height, width, k, 2)
    """

    # -- prepare data --
    c, h, w = img_shape
    nframes = burst.shape[0]
    burstPad = padBurst(burst,img_shape,patchsize,nblocks)
    burst_ptr,burst_type = getImage(burstPad)
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,nframes)
    bl,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,
                                       device,is_tensor,nframes)
    weights_ptr = torch2swig(weights)
    # print("bl")
    # print("-"*50)
    # for i in range(bl.shape[1]):
    #     print(bl[:,i,:].cpu().numpy())
    # print(bl.shape)
    # print("-"*50)
    # print("bl")
    
    # -- setup args --
    args = faiss.GpuWeightedBurstNnfDistanceParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.t = nframes
    args.ps = patchsize
    args.nblocks = nblocks
    args.nblocks_total = bl.shape[1]
    args.valMean = valMean # noise level value offset of minimum
    args.weights = weights_ptr
    args.burst = burst_ptr
    args.dtype = burst_type
    args.blockLabels = blockLabels_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        with using_stream(res):
            faiss.bfBurstNnf(res, args)
    else:
        faiss.bfBurstNnf(res, args)

    return vals, locs
