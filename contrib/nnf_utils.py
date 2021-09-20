###########################################
# NNF GPU Functions
###########################################

import torch
import faiss
import numpy as np
from torch_utils import swig_ptr_from_UInt8Tensor,swig_ptr_from_HalfTensor,swig_ptr_from_FloatTensor,swig_ptr_from_IntTensor,swig_ptr_from_IndicesTensor

# -- helpers --

def get_optional_device(array):
    if torch.is_tensor(array):
        return array.device
    else:
        return None

def torch2swig(tensor):
    if tensor.dtype == torch.int64:
        tensor_ptr = swig_ptr_from_IndicesTensor(tensor)
        tensor_dtype = faiss.IndicesDataType_I64
    elif tensor.dtype == torch.int32:
        tensor_ptr = swig_ptr_from_IntTensor(tensor)
        tensor_dtype = faiss.IndicesDataType_I32
    elif tensor.dtype == torch.float32:
        tensor_ptr = swig_ptr_from_FloatTensor(tensor)
        tensor_type = faiss.DistanceDataType_F32
    elif tensor.dtype == torch.float16:
        tensor_ptr = swig_ptr_from_HalfTensor(tensor)
        tensor_type = faiss.DistanceDataType_F16
    else:
        raise KeyError("Uknown tensor dtype.")
    return tensor_ptr,tensor_dtype

#---------------------------------
#    
#      Pad the Image
#    
#---------------------------------

def padImage(img,img_shape,patchsize,nblocks):

    # -- init --
    dims = len(img_shape)
    assert dims == 3,"Our image is c, h, w."
    th_pad = torch.nn.functional.pad
    is_tensor = torch.is_tensor(img)

    # -- create padded shape --
    pad = patchsize//2 + nblocks//2
    padded_shape = [img_shape[i] for i in range(dims)]
    padded_shape[1] += 2*pad
    padded_shape[2] += 2*pad
    is_padded = np.all([padded_shape[i] == img.shape[i] for i in range(dims)])
    print(padded_shape)
    print(img.shape)
    
    if not(is_padded):

        # -- convert to tensor if necessary --
        if not(is_tensor): img = torch.Tensor(img)
    
        # -- add padding to image --
        pads = (pad,)*4
        padded_img = th_pad(img[None,:],pads,mode="reflect")[0]
    
        # -- convert back to numpy if necessary --
        if not(is_tensor): img = img.numpy()

    else:
        padded_img = img

    for d in range(dims):
        msg = "{} v {}".format(padded_img.shape[d],padded_shape[d])
        msg = "We just padded tho... {}".format(msg)
        assert np.all(padded_img.shape[d] == padded_shape[d]),msg
    return padded_img

#---------------------------------
#    
#   Get Block Labels for NNF 
#    
#---------------------------------

def getBlockLabels(blockLabels,nblocks,dtype,device,is_tensor):
    blockLabels = getBlockLabelsNumpy(blockLabels,nblocks,dtype)
    if not(torch.is_tensor(blockLabels)) and is_tensor:
        if device != 'cpu':
            blockLabels = torch.cuda.IntTensor(blockLabels,device=device)
        else:
            blockLabels = torch.IntTensor(blockLabels,device=device)
        blockLabels_ptr,_ = torch2swig(blockLabels)
    else:
        blockLabels_ptr = faiss.swig_ptr(blockLabels)
    return blockLabels,blockLabels_ptr

def getBlockLabelsNumpy(blockLabels,nblocks,dtype):
    if blockLabels is None:
        blockLabels = np.zeros((nblocks**2,2))
        blockLabelsTmp = np.arange(nblocks**2).reshape(nblocks,nblocks)
        for i in range(nblocks**2):
            x,y = np.where(blockLabelsTmp == i)
            blockLabels[i,0] = x
            blockLabels[i,1] = y
    return blockLabels.astype(np.int32)

#---------------------------------
#    
#     Prepare Index Data
#    
#---------------------------------

def getVals(vals,h,w,k,device,is_tensor):
    if is_tensor:
        return getValsTensor(vals,h,w,k,device)
    else:
        return getValsNumpy(vals,h,w,k)

def getValsTensor(vals,h,w,k,device):
    if D is None:
        D = torch.empty(h, w, k, device=device, dtype=torch.float32)
    else:
        assert D.shape == (h, w, k)
        # interface takes void*, we need to check this
        assert (D.dtype == torch.float32)
    vals_ptr,_ = tensor2swig(vals)
    return vals,vals_ptr

def getValsNumpy(vals,h,w,k):
    if vals is None:
        vals = np.empty((h, w, k), dtype=np.float32)
    else:
        assert vals.shape == (h, w, k)
        # interface takes void*, we need to check this
        assert vals.dtype == np.float32
    vals_ptr = faiss.swig_ptr(vals)
    return vals,vals_ptr

#---------------------------------
#    
#     Prepare Index Data
#    
#---------------------------------

def getLocs(locs,h,w,k,device,is_tensor):
    if is_tensor:
        return getLocsTorch(locs,h,w,k,device)
    else:
        return getLocsNumpy(locs,h,w,k)

def getLocsTorch(locs,h,w,k,device):
    if locs is None:
        locs = torch.empty(h, w, k, 2, device=device, dtype=torch.int32)
    else:
        assert locs.shape == (h, w, k, 2)
    locs_ptr,locs_type = torch2swig(tensor)
    return locs_ptr,locs_type,faiss.IndicesDataType_I64

def getLocsNumpy(locs,h,w,k):
    if locs is None:
        locs = np.empty((h, w, k, 2), dtype=np.int32)
    else:
        assert locs.shape == (h, w, k, 2)

    locs_ptr = faiss.swig_ptr(locs)

    if locs.dtype == np.int64:
        locs_type = faiss.IndicesDataType_I64
    elif locs.dtype == locs.dtype == np.int32:
        locs_type = faiss.IndicesDataType_I32
    else:
        raise TypeError('I must be i64 or i32')
    return locs,locs_ptr,locs_type


#---------------------------------
#    
#     Prepare Image Data
#    
#---------------------------------


def getImage(img):
    if torch.is_tensor(img):
        return getImageTorch(img)
    else:
        return getImageNumpy(img)

def getImageTorch(img):
    c, h, w = img.size()
    if not(img.is_contiguous()):
        raise TypeError('matrix should be row or column-major')
    if img.dtype == torch.float32:
        img_ptr = swig_ptr_from_FloatTensor(img)
        img_type = faiss.DistanceDataType_F32
    elif img.dtype == torch.float16:
        img_ptr = swig_ptr_from_HalfTensor(img)
        img_type = faiss.DistanceDataType_F16
    else:
        raise TypeError('img must be f32 or f16')
    return img_ptr,img_type

def getImageNumpy(img):
    c, h, w = img.shape
    if not img.flags.c_contiguous:
        raise TypeError('img matrix should be row (C) or column-major (Fortran)')

    img_ptr = faiss.swig_ptr(img)

    if img.dtype == np.float32:
        img_type = faiss.DistanceDataType_F32
    elif img.dtype == np.float16:
        img_type = faiss.DistanceDataType_F16
    else:
        raise TypeError('img must be float32 or float16')
    return img_ptr,img_type



def runNnfBurst(burst, patchsize, nblocks, k = 3,
                valMean = 0., blockLabels=None, ref_t=None):

    # -- setup res --
    res = faiss.StandardGpuResources()

    # -- set padded images --
    nframes,nimages,c,h,w = burst.shape
    device = burst.device
    img_shape = (c,h,w)
    if ref_t is None: ref_t = nframes // 2
    blockLabels,_ = getBlockLabels(blockLabels,nblocks,np.int32,device,is_tensor)
    
    # -- run nnf --
    for i in range(nimages):
        refImg = burst[ref_t,i]
        refImgPad = padImage(refImg,img_shape,patchsize,nblocks)    
        for t in range(nframes):
            if t == ref_t:
                vals_t = torch.zeros((h,w,k))
                locs_t = torch.zeros((h,w,k,2))
            else:
                tgtImg = burst[t,i]
                vals_t,locs_t = runNnf(res, img_shape, refImg,
                                       tgtImg, None, None,
                                       k, patchsize, nblocks,
                                       valMean = 0., blockLabels=blockLabels)
            vals.append(vals_t)
            locs.append(locs_t)
    vals = torch.stack(vals)
    locs = torch.stack(locs)

    return vals,locs

def runNnf(res, img_shape, refImg, tgtImg, vals, locs, patchsize, nblocks, k = 3, valMean = 0., blockLabels=None):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    ref : array_like
        Reference image, shape (c, h, w).
        `dtype` must be float32.
    target : array_like
        Target image, shape (c, h, w).
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
    refImgPad = padImage(refImg,img_shape,patchsize,nblocks)
    tgtImgPad = padImage(tgtImg,img_shape,patchsize,nblocks)
    refImg_ptr,refImg_type = getImage(refImgPad)
    tgtImg_ptr,tgtImg_type = getImage(tgtImgPad)
    is_tensor = torch.is_tensor(refImg)
    device = get_optional_device(refImg)
    assert torch.is_tensor(refImg) == torch.is_tensor(tgtImg),"Both torch or numpy."
    assert tgtImg_type == refImg_type,"Only one type for both"
    vals,vals_ptr = getVals(vals,h,w,k,device,is_tensor)
    locs,locs_ptr,locs_type = getLocs(locs,h,w,k,device,is_tensor)
    _,blockLabels_ptr = getBlockLabels(blockLabels,nblocks,locs.dtype,device,is_tensor)

    # -- setup args --
    args = faiss.GpuNnfDistanceParams()
    args.metric = faiss.METRIC_L2
    args.k = k
    args.h = h
    args.w = w
    args.c = c
    args.ps = patchsize
    args.nblocks = nblocks
    args.valMean = valMean # noise level value offset of minimum
    args.dtype = refImg_type
    args.refImg = refImg_ptr
    args.targetImg = tgtImg_ptr
    args.blockLabels = blockLabels_ptr
    args.outDistances = vals_ptr
    args.outIndices = locs_ptr
    args.outIndicesType = locs_type
    args.ignoreOutDistances = True

    # -- choose to block with or without stream --
    if is_tensor:
        with using_stream(res):
            faiss.bfNnf(res, args)
    else:
        faiss.bfNnf(res, args)

    return vals, locs
