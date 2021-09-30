import torch
import faiss
import numpy as np
from einops import rearrange,repeat
from torch_utils import swig_ptr_from_UInt8Tensor,swig_ptr_from_HalfTensor,swig_ptr_from_FloatTensor,swig_ptr_from_IntTensor,swig_ptr_from_IndicesTensor,swig_ptr_from_BoolTensor

# -- helpers --

def get_optional_device(array):
    if torch.is_tensor(array):
        return array.device
    else:
        return None

def flip_array_like(array,dim):
    if torch.is_tensor(array):
        array = torch.flip(array,(dim,)).clone().contiguous()
    else:
        array = np.flip(array,dim).copy()
        array = np.ascontiguousarray(array)
    return array

def torch2swig(tensor):
    if tensor.dtype == torch.int64:
        tensor_ptr = swig_ptr_from_IndicesTensor(tensor)
        tensor_dtype = faiss.IndicesDataType_I64
    elif tensor.dtype == torch.int32:
        tensor_ptr = swig_ptr_from_IntTensor(tensor)
        tensor_dtype = faiss.IndicesDataType_I32
    elif tensor.dtype == torch.float32:
        tensor_ptr = swig_ptr_from_FloatTensor(tensor)
        tensor_dtype = faiss.DistanceDataType_F32
    elif tensor.dtype == torch.float16:
        tensor_ptr = swig_ptr_from_HalfTensor(tensor)
        tensor_dtype = faiss.DistanceDataType_F16
    elif tensor.dtype == torch.bool:
        tensor_ptr = swig_ptr_from_BoolTensor(tensor)
        tensor_dtype = None # not needed
    else:
        raise KeyError("Uknown tensor dtype.")
    return tensor_ptr,tensor_dtype

def get_swig_ptr(th_or_np):
    if torch.is_tensor(th_or_np):
        array_ptr,_ = torch2swig(th_or_np)
    else:
        array_ptr = faiss.swig_ptr(th_or_np)
    return array_ptr

#---------------------------------
#    
#  Simulate Chi^2 to Approx Mode
#    
#---------------------------------


#---------------------------------
#    
#      Pad the Image
#    
#---------------------------------

def padBurst(burst,img_shape,patchsize,nblocks):

    # -- check if already padded --
    dims = len(img_shape)
    assert dims == 3,"Our image is c, h, w."
    pad = patchsize//2 + nblocks//2
    padded_shape = [img_shape[i] for i in range(dims)]
    padded_shape[1] += 2*pad
    padded_shape[2] += 2*pad
    is_padded = True
    for img in burst:
        eq_size = np.all([padded_shape[i] == img.shape[i] for i in range(dims)])
        if not(eq_size):
            is_padded = False
            break

    # -- add padding --
    if not(is_padded):
        burstPad = []
        for img in burst:
            imgPad = padImage(img,img_shape,patchsize,nblocks)
            burstPad.append(imgPad)
        burstPad = torch.stack(burstPad)
        return burstPad
    else:
        return burst
    
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

def create_meshgrid_blockLabels(blockLabels,nframes):
    nblocks2 = blockLabels.shape[0]
    coord_1d = np.arange(nblocks2)
    to_mesh = [coord_1d,]*nframes
    to_mesh[nframes//2] = [[nblocks2//2]]
    mesh = np.stack(np.meshgrid(*to_mesh)).reshape((nframes,-1))
    blockLabelsMesh = []
    for idx in range(mesh.shape[1]):
        mesh_i = mesh[:,idx]
        labels_i = []
        for t in range(nframes):
            label = blockLabels[mesh_i[t]]
            labels_i.append(label)
        blockLabelsMesh.append(labels_i)
    blockLabelsMesh = np.stack(blockLabelsMesh,axis=1)
    return blockLabelsMesh

def getMask(nsearch,h,w,sub_nframes,device,is_tensor):
    mask = torch.ones(nsearch,h,w,sub_nframes).to(device).type(torch.bool)
    mask_ptr = get_swig_ptr(mask)
    return mask,mask_ptr

def getBlockLabelsFull(blockLabels,ishape,nblocks,dtype,device,is_tensor,t=None):
    blLabels,_ = getBlockLabels(blockLabels,nblocks,dtype,device,is_tensor,t)
    c,h,w = ishape
    blLabels = repeat(blLabels,'t l two -> l h w t two',h=h,w=w)
    blockLabels_ptr = get_swig_ptr(blLabels)
    return blLabels,blockLabels_ptr

def getBlockLabels(blockLabels,nblocks,dtype,device,is_tensor,t=None):
    blockLabels = getBlockLabelsNumpy(blockLabels,nblocks,dtype,t)
    blockLabels = flip_array_like(blockLabels,1)
    if not(torch.is_tensor(blockLabels)) and is_tensor:
        if device != 'cpu':
            blockLabels = torch.cuda.IntTensor(blockLabels,device=device)
        else:
            blockLabels = torch.IntTensor(blockLabels,device=device)
    blockLabels_ptr = get_swig_ptr(blockLabels)
    return blockLabels,blockLabels_ptr

def getBlockLabelsNumpy(blockLabels,nblocks,dtype,t=None):
    if blockLabels is None:
        blockLabels = getBlockLabelsRaw(nblocks)
        if not(t is None): # meshgrid for t == 1 is the same as no meshgrid.
            blockLabels = create_meshgrid_blockLabels(blockLabels,t)
            # blockLabels = repeat(blockLabels,'b two -> t b two',t=t)
    if isinstance(blockLabels,np.ndarray):
        return blockLabels.astype(np.int32)
    elif torch.is_tensor(blockLabels):
        return blockLabels.type(torch.int32)
    else:
        return blockLabels

def getBlockLabelsRaw(nblocks):
    blockLabels = np.zeros((nblocks**2,2))
    blockLabelsTmp = np.arange(nblocks**2).reshape(nblocks,nblocks)
    for i in range(nblocks**2):
        x,y = np.where(blockLabelsTmp == i)
        blockLabels[i,0] = x
        blockLabels[i,1] = y
    blockLabels -= nblocks//2
    return blockLabels

#---------------------------------
#    
#     Prepare Index Data
#    
#---------------------------------

def getVals(vals,h,w,k,device,is_tensor,t=None):
    if is_tensor:
        return getValsTensor(vals,h,w,k,device,t)
    else:
        return getValsNumpy(vals,h,w,k,t)

def getValsTensor(vals,h,w,k,device,t=None):
    # -- get shape --
    if t is None: shape = (h, w, k)
    else: shape = (t, h, w, k)

    # -- create or assert --
    if vals is None:
        vals = torch.ones(*shape, device=device, dtype=torch.float32)
        vals *= torch.finfo(torch.float32).max
    else:
        assert vals.shape == shape
        # interface takes void*, we need to check this
        assert (vals.dtype == torch.float32)
    vals_ptr,_ = torch2swig(vals)
    return vals,vals_ptr

def getValsNumpy(vals,h,w,k,t=None):

    # -- get shape --
    if t is None: shape = (h, w, k)
    else: shape = (t, h, w, k)

    # -- create or assert --
    if vals is None:
        vals = np.ones(shape, dtype=np.float32)
        vals *= np.finfo(np.float32).max
    else:
        assert vals.shape == shape
        # interface takes void*, we need to check this
        assert vals.dtype == np.float32
    vals_ptr = faiss.swig_ptr(vals)
    return vals,vals_ptr

#---------------------------------
#    
#     Prepare Index Data
#    
#---------------------------------

def getLocs(locs,h,w,k,device,is_tensor,t=None):
    if is_tensor:
        return getLocsTorch(locs,h,w,k,device,t)
    else:
        return getLocsNumpy(locs,h,w,k,t)

def getLocsTorch(locs,h,w,k,device,t=None):
    if locs is None:
        if t is None:
            locs = torch.empty(h, w, k, 2,
                               device=device, dtype=torch.int32)
        else:
            locs = torch.empty(t, h, w, k, 2,
                               device=device, dtype=torch.int32)
    else:
        if t is None: assert locs.shape == (h, w, k, 2)
        else: assert locs.shape == (t, h, w, k, 2)            
    locs_ptr,locs_type = torch2swig(locs)
    return locs,locs_ptr,locs_type

def getLocsNumpy(locs,t,h,w,k):
    if locs is None:
        locs = np.empty((t, h, w, k, 2), dtype=np.int32)
    else:
        assert locs.shape == (t, h, w, k, 2)

    locs_ptr = faiss.swig_ptr(locs)

    if locs.dtype == np.int64:
        locs_type = faiss.IndicesDataType_I64
    elif locs.dtype == locs.dtype == np.int32:
        locs_type = faiss.IndicesDataType_I32
    else:
        raise TypeError('locs must be i64 or i32')
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


