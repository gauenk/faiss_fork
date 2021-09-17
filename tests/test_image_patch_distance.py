import time
import torch
import faiss
import contextlib
import numpy as np
from einops import rearrange
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from align import nnf 

def swig_ptr_from_FloatTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    # assert x.is_contiguous()
    assert x.dtype == np.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

@contextlib.contextmanager
def using_stream(res, pytorch_stream=None):
    """ Creates a scoping object to make Faiss GPU use the same stream
        as pytorch, based on torch.cuda.current_stream().
        Or, a specific pytorch stream can be passed in as a second
        argument, in which case we will use that stream.
    """

    if pytorch_stream is None:
        pytorch_stream = torch.cuda.current_stream()

    # This is the cudaStream_t that we wish to use
    cuda_stream_s = faiss.cast_integer_to_cudastream_t(pytorch_stream.cuda_stream)

    # So we can revert GpuResources stream state upon exit
    prior_dev = torch.cuda.current_device()
    prior_stream = res.getDefaultStream(torch.cuda.current_device())

    res.setDefaultStream(torch.cuda.current_device(), cuda_stream_s)

    # Do the user work
    try:
        yield
    finally:
        res.setDefaultStream(prior_dev, prior_stream)


def pad_numpy(ndarray,pads):
    dtype = ndarray.dtype
    tensor = torch.Tensor(ndarray)
    padded = torch.nn.functional.pad(tensor[None,:],pads)[0]
    ndpadded = np.array(padded,dtype=dtype)
    return ndpadded

# -- settings --


# h,w,c = 1024,4096,3
# h,w,c = 32,32,3
# h,w,c = 16,16,3
# h,w,c = 17,17,3
# h,w,c = 256,256,3
h,w,c = 1024,1024,3
# h,w,c = 128,128,3
k,ps,nblocks = 9,3,3
pad = int(ps//2)
pads = (pad,pad,pad,pad)


# -- create blockLabels --
blockLabels = np.zeros((nblocks**2,2))
blockLabelsTmp = np.arange(nblocks**2).reshape(nblocks,nblocks)
for i in range(nblocks**2):
    x,y = np.where(blockLabelsTmp == i)
    blockLabels[i,0] = x
    blockLabels[i,1] = y
# blockLabels -= 1
print(blockLabels - 1)

# -- create images --
ref_image = np.random.rand(c,h,w).astype(np.float32)
bfnnf_ref_image = pad_numpy(ref_image,pads)
bfnnf_ref_image = np.ascontiguousarray(bfnnf_ref_image)
ref_image_T = torch.Tensor(ref_image)
target_image_T = torch.nn.functional.pad(ref_image_T[None,:,1:,:-1],(1+pad,pad,pad,1+pad))[0]
# target_image_T = torch.nn.functional.pad(ref_image_T[None,:,:-1,1:],(pad,1+pad,pad,1+pad))[0]
target_image = np.copy(np.array(target_image_T))
target_image = np.ascontiguousarray(target_image)
print("target_image.shape ",target_image.shape)


#
# -- compute FAISS nnf --
#

# -- benchmark runtime --

nnf_ref = torch.Tensor(ref_image)
# nnf_target = torch.nn.functional.pad(ref_image_T[None,:,:-1,1:],(1,1,0,0))[0]
# nnf_target = torch.nn.functional.pad(ref_image_T[None,:,1:,1:],(0,1,1,0))[0]
# nnf_target = torch.nn.functional.pad(torch.Tensor(target_image)[None,:],(1,1,1,1))[0]
nnf_target = target_image[:,pad:nnf_ref.shape[1]+pad,pad:nnf_ref.shape[2]+pad]#ref_image_T
nnf_target = torch.Tensor(np.copy(np.array(nnf_target)))
start_time = time.perf_counter()
nnf_vals,nnf_locs = nnf.compute_nnf(nnf_ref,nnf_target,ps,K=9,gpuid=0)
nnf_runtime = time.perf_counter() - start_time

# -- get output comparison --

# nnf_ref = torch.Tensor(ref_image)
# nnf_target = torch.nn.functional.pad(ref_image_T[None,:,1:,1:],(1,1,0,0))[0]
# nnf_target = torch.Tensor(np.copy(np.array(nnf_target)))
# start_time = time.perf_counter()
# nnf_vals,nnf_locs = nnf.compute_nnf(nnf_ref,nnf_target,ps,K=9,gpuid=0)

# target_image = np.ascontiguousarray(np.random.rand(h+2*pad,w+2*pad,c)).astype(np.float32)

vals = np.ascontiguousarray(np.zeros((h,w,k))).astype(np.float32)
locs = np.ascontiguousarray(np.zeros((h,w,k,2))).astype(np.int32)


# swig_target_image = swig_ptr_from_FloatTensor(target_image)
# swig_ref_image = swig_ptr_from_FloatTensor(ref_image)
# swig_val = swig_ptr_from_FloatTensor(vals)
# swig_locs = swig_ptr_from_FloatTensor(locs)

# -- create gpu resources --
print("torch cuda set_device")
torch.cuda.set_device(0)
print("[pre] res")
res = faiss.StandardGpuResources()
print("[post] res")
# res.setDefaultNullStreamAllDevices()
# res.setDefaultStream(0,0)
# print(dir(res))
# print(dir(faiss))
# print(faiss.get_num_gpus())
# print(res.getMemoryInfo())
# print(res.getResources())
# print(res.getDefaultStream(0))
# print(res.getTempMemoryAvailable(2))
# print(res.getDefaultStream(0))
# exit()

# -- numpy to swig --
swig_ref_image = faiss.swig_ptr(bfnnf_ref_image)
swig_target_image = faiss.swig_ptr(target_image)
swig_vals = faiss.swig_ptr(vals)
swig_locs = faiss.swig_ptr(locs)
swig_blockLabels = faiss.swig_ptr(blockLabels.astype(np.int32))
print(type(swig_vals))
print("bfnnf_ref_image.shape ",bfnnf_ref_image.shape)
print("target_image.shape ",target_image.shape)

# -- error checking --
assert bfnnf_ref_image.shape == target_image.shape, "same shape."
assert np.sum(vals.shape[0] - (bfnnf_ref_image.shape[1]-2*pad)) == 0, "same shape."
assert np.sum(vals.shape[1] - (bfnnf_ref_image.shape[2]-2*pad)) == 0, "same shape."
assert np.sum(locs.shape[0] - (bfnnf_ref_image.shape[1]-2*pad)) == 0, "same shape."
assert np.sum(locs.shape[1] - (bfnnf_ref_image.shape[2]-2*pad)) == 0, "same shape."

# -- create arguments --
args = faiss.GpuNnfDistanceParams()
args.metric = faiss.METRIC_L2
args.k = k
args.h = h
args.w = w
args.c = c
args.ps = ps
args.nblocks = nblocks
args.dtype = faiss.DistanceDataType_F32
args.targetImg = swig_target_image
args.targetImageType = faiss.DistanceDataType_F32
args.refImg = swig_ref_image
args.refImageType = faiss.DistanceDataType_F32
args.refPathNorms = None
args.blockLabels = swig_blockLabels
args.outDistances = swig_vals
args.ignoreOutDistances = True
args.outIndicesType = faiss.IndicesDataType_I32
args.outIndices = swig_locs

# -- call function --
print("Starting bfNnf")
start_time = time.perf_counter()
faiss.bfNnf(res, args)
bfNnf_runtime = time.perf_counter() - start_time

# print(vals)
print(ref_image[:,:3,:3])
print(vals.shape)
print("cuda vals!")
print("FAISS runtime [\"Unfold\" + Global Search]: ",nnf_runtime)
print("Our-L2 runtime [No \"Unfold\" + Local Search]: ",bfNnf_runtime)
print("Our-L2 Output: ",vals[4,4,:])

blockLabelsInt = (blockLabels - 1).astype(np.int32)
xstart,ystart = 3,3
res = []
for i in range(nblocks**2):
    x,y = blockLabelsInt[i,:]
    ref_xy = nnf_ref[:,xstart+pad:xstart+ps+pad,ystart+pad:ystart+ps+pad]
    tgt_xy = nnf_target[:,xstart+x+pad:xstart+x+ps+pad,ystart+y+pad:ystart+y+ps+pad]
    res.append(float(torch.sum(torch.pow( ref_xy - tgt_xy , 2)).item()))
print("Expected Output: ",np.array(res))

# print("nnf vals!")
# for i in range(3):
#     print(nnf_vals[0,0,1,i,:].astype(np.float32))

