
"""
Python calling the C++ API
for KMeans+Burst Search

"""

def runKmBurstSearch(burst, patchsize, nblocks, k=1,
                     kmeansK = 3, std = None, traj=None,
                     img_shape=None, fmt=False):
    
    # -- create faiss GPU resource --
    res = faiss.StandardGpuResources()

    # -- get shapes for low-level exec of FAISS --
    nframes,nimages,c,h,w = burst.shape
    if img_shape is None: img_shape = [c,h,w]
    device = burst.device
    if ref is None: ref = nframes//2
    if std is None: std = 0.

    # -- compute search across image burst --
    vals,locs = [],[]
    for i in range(nimages):

        # -- create padded burst --
        burstPad_i = padBurst(burst[:,i],img_shape,patchsize,nblocks)

        # -- assign input vals and locs --
        traj_i = traj
        if not(traj is None): traj_i = traj[i]

        # -- execute over search space! --
        vals_i,locs_i = _runKmBurstSearch(res, burstPad_i, patchsize, nblocks,
                                          k = k, kmeansK = kmeansK,
                                          img_shape = img_shape, 
                                          std = std, ref = ref, traj=traj_i)

        vals.append(vals_i)
        locs.append(locs_i)
    vals = torch.stack(vals,dim=0)
    # (nimages, h, w, k)
    locs = torch.stack(locs,dim=1)
    # (nframes, nimages, h, w, k, two)

    if to_flow:
        locs_y = locs[...,0]
        locs_x = locs[...,1]
        locs = torch.stack([locs_x,-locs_y],dim=-1)
    
    if fmt:
        vals = rearrange(vals,'i h w k -> i (h w) k').cpu()
        locs = rearrange(locs,'t i h w k two -> k i (h w) t two').cpu().long()

    return vals,locs

def _runKmBurstSearch(res, burst , patchsize, nblocks,
                      k = 1, kmeansK = 3, img_shape = None,
                      std = None, ref = None, traj=None):
    
    

    # ----------------------
    #
    #    init none vars
    #
    # ----------------------

    nframes,c,h,w = burst.shape
    if img_shape is None: img_shape = [c,h,w]
    if ref is None: ref = nframes//2
    if traj is None: traj = compute_zero_traj(nblocks)
    if std is None: raise ValueError("Uknown std -- must be a float.")
    c, h, w = img_shape
    is_tensor = torch.is_tensor(burst)
    device = get_optional_device(burst)

    # ----------------------
    #
    #     prepare data
    #
    # ----------------------

    burstPad = padBurst(burst,img_shape,patchsize,nblocks)
    burst_ptr,burst_type = getImage(burstPad)
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor,None)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor,nframes)
    bl,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,
                                       device,is_tensor,nframes)
    
    # -- setup args --
    args = faiss.GpuKmBurstParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.t = nframes
    args.ps = patchsize
    args.nblocks = nblocks
    args.nblocks_total = bl.shape[1]
    args.kmeansK = kmeansK
    args.std = std # noise level
    args.burst = burst_ptr
    args.dtype = burst_type
    args.traj = traj_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        with using_stream(res):
            faiss.bfKmBurst(res, args)
    else:
        faiss.bfKmBurst(res, args)

    return vals, locs

