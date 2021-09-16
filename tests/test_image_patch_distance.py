import torch
import faiss
import contextlib
import numpy as np

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


# -- settings --

# h,w,c = 1024,4096,3
h,w,c = 32,32,3
k,ps,nblocks = 2,7,3
pad = int(ps//2)
ref_image = np.ascontiguousarray(np.random.rand(h,w,c)).astype(np.float32)
target_image = np.ascontiguousarray(np.random.rand(h+2*pad,w+2*pad,c)).astype(np.float32)
vals = np.ascontiguousarray(np.zeros((h,w,k))).astype(np.float32)
locs = np.ascontiguousarray(np.zeros((h,w,k,2))).astype(np.int32)

# -- numpy to swig --
swig_ref_image = faiss.swig_ptr(ref_image)
swig_target_image = faiss.swig_ptr(target_image)
swig_vals = faiss.swig_ptr(vals)
swig_locs = faiss.swig_ptr(locs)
print(type(swig_vals))

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
args.outDistances = swig_vals
args.ignoreOutDistances = True
args.outIndicesType = faiss.IndicesDataType_I32
args.outIndices = swig_locs

# -- call function --
print("Starting bfNnf")
faiss.bfNnf(res, args)

print(vals)
