import time
import torch
import faiss
import contextlib
import numpy as np
from einops import rearrange
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")
from align import nnf 
from align.xforms import pix_to_blocks

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
h,w,c = 512,512,3
# h,w,c = 256,256,3
# h,w,c = 32,32,3
# h,w,c = 48,48,3
# h,w,c = 32,32,3
# h,w,c = 1024,1024,3
# h,w,c = 32,32,3
# ps,nblocks = 11,10
ps,nblocks = 3,3
k = 2
pad = int(ps//2)+int(nblocks//2)
pads = (pad,pad,pad,pad)


# -- create blockLabels --
blockLabels = np.zeros((nblocks**2,2))
blockLabelsTmp = np.arange(nblocks**2).reshape(nblocks,nblocks)
for i in range(nblocks**2):
    x,y = np.where(blockLabelsTmp == i)
    blockLabels[i,0] = x
    blockLabels[i,1] = y
blockLabels -= (nblocks//2)
# blockLabels = blockLabels[[4,5],:]
# k,nblocks = 4,2
# print(blockLabels - 1)

# -- create images --
ref_image = np.random.rand(c,h,w).astype(np.float32)
# ref_image[0,:,:] = 0
# ref_image[1,:,:] = 1.
bfnnf_ref_image = pad_numpy(ref_image,pads)
bfnnf_ref_image = np.ascontiguousarray(bfnnf_ref_image)
ref_image_T = torch.Tensor(ref_image)
target_image_T = torch.nn.functional.pad(ref_image_T[None,:,:-1,:-1],(1+pad,pad,1+pad,pad),mode="reflect")[0]
bfnnf_ref_image = torch.nn.functional.pad(ref_image_T[None,:,2:,:-2],(pad,2+pad,pad,2+pad),mode="reflect")[0]
bfnnf_ref_image = np.copy(np.array(bfnnf_ref_image))
# bfnnf_ref_image = np.copy(np.array(target_image_T))
bfnnf_ref_image = np.ascontiguousarray(bfnnf_ref_image)

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
# nnf_target = target_image[:,pad:nnf_ref.shape[1]+pad,pad:nnf_ref.shape[2]+pad]#ref_image_T
# nnf_target = target_image[:,:nnf_ref.shape[1],:nnf_ref.shape[2]]#ref_image_T
nnf_ref = torch.Tensor(bfnnf_ref_image)
# nnf_target = target_image_T[:,:nnf_ref.shape[1],:nnf_ref.shape[2]]
nnf_target = target_image
nnf_target = torch.Tensor(np.copy(np.array(nnf_target)))

# pad = ps//2 + nblocks
nnf_ref = nnf_ref[:,pad:-pad,pad:-pad]
nnf_target = nnf_target[:,pad:-pad,pad:-pad]
nnf_vals,nnf_locs = nnf.compute_nnf(nnf_ref,nnf_target,ps,K=k,gpuid=0)
time.sleep(3)
start_time = time.perf_counter()
nnf_vals,nnf_locs = nnf.compute_nnf(nnf_ref,nnf_target,ps,K=k,gpuid=0)
nnf_runtime = time.perf_counter() - start_time

# -- get output comparison --

print(nnf_locs[0,0,0,0,0])
print(nnf_locs[0,0,-1,-1,0])
print(nnf_locs[0,0,9,9,0])

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
bfnnf_ref_image = np.ascontiguousarray(bfnnf_ref_image)
target_image = np.ascontiguousarray(target_image)
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
print( blockLabels.shape )


# -- create arguments --
args = faiss.GpuNnfDistanceParams()
args.metric = faiss.METRIC_L2
args.k = k
args.h = h
args.w = w
args.c = c
args.ps = ps
args.nblocks = nblocks
args.valMean = 0. # noise level value offset of minimum
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
# faiss.bfNnf(res, args)
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
import nnf_utils as nnf_utils
refImg = bfnnf_ref_image
tgtImg = target_image
vals,locs = None,None
patchsize = ps
blockLabels = blockLabels
shape = (c,h,w)
vals,locs = nnf_utils.runNnf(res, shape, refImg, tgtImg, vals, locs,
                             patchsize, nblocks, k,
                             valMean = 0., blockLabels=blockLabels)
bfNnf_runtime = time.perf_counter() - start_time
# print(locs[:3,:3])

# print(vals)
# print(ref_image[:,:3,:3])
print(vals.shape)
print("cuda vals!")
print("FAISS runtime [\"Unfold\" + Global Search]: ",nnf_runtime)
print("Our-L2 runtime [No \"Unfold\" + Local Search]: ",bfNnf_runtime)

check_xy = [0,0]
print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
check_xy = [1,1]
print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
check_xy = [h//2,w//2]
print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
check_xy = [-2,-2]
print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
check_xy = [-1,-1]
print("Our-L2 Output: ",vals[check_xy[0],check_xy[1],:])
print("Zero Check: ",np.any(vals==0))
print("Nan Check: ",np.any(np.isnan(vals)))

# offset = ps//2
print(nnf_locs[0,0,h//2,w//2,0])
# nnf_locs_x = nnf_locs[0,0,offset:-offset,offset:-offset,0,1] \
#     - np.arange(w)[None,:] - offset
# nnf_locs_y = nnf_locs[0,0,offset:-offset,offset:-offset,0,0] \
#     - np.arange(h)[:,None] - offset
nnf_locs_x = nnf_locs[0,0,:,:,0,1] \
    - np.arange(w)[None,:]
nnf_locs_y = nnf_locs[0,0,:,:,0,0] \
    - np.arange(h)[:,None]
nnf_locs = np.stack([nnf_locs_y,nnf_locs_x],axis=-1)
intIdx = slice(nblocks//2+ps//2,-(nblocks//2+ps//2))
print("NNF vs Ours [locs]: ", np.sum(np.abs(locs[intIdx,intIdx,0] -\
                                            nnf_locs[intIdx,intIdx])))
print(locs[h//2:h//2+2,w//2:w//2+2,0])
print(nnf_locs[h//2:h//2+2,w//2:w//2+2])

# nnf_locs_to_xform = rearrange(nnf_locs[0,0,1:-1,1:-1,0],'h w k -> 1 (h w) 1 k')
# nnf_pix = pix_to_blocks(torch.LongTensor(nnf_locs_to_xform),nblocks)
# nnf_pix = rearrange(nnf_pix,'1 (h w) 1 -> h w',h=h)
# print(nnf_pix[8,8])

# print(vals)
# print(locs)

# for i in range(vals.shape[1]):
#     print(i,vals[i,:],vals[i,:].shape)
# print("vals.")
# print(vals[8,8,:])
# print("nnf.")
# for i in [6,7,8,9]:
#     for j in [6,7,8,9]:
#         print(nnf_vals[0,0,i,j,:])
# print(nnf_vals[0,0,9,9,:])
# print(nnf_vals[0,0,10,10,:])
# print(nnf_vals[0,0,:,:,1])

# for i in range(-5,5,1):
#     print("L2-Output: ",vals[check_xy[0]+i,check_xy[1]+i,:].astype(np.float32))

# for i in range(5):
#     print("NNF Output: ",nnf_vals[0,0,check_xy[0]+i,check_xy[1]+i,:].astype(np.float32))


#
# -- Compare Outputs --
# 


#
# --- Version 1 --
#

# blockLabelsInt = (blockLabels - 1).astype(np.int32)
# print(np.sum(np.abs(bfnnf_ref_image[:,1,1] - ref_image[:,0,0])))
# print(np.sum(np.abs(target_image[:,1,1] - nnf_target[:,0,0].numpy())))
blockLabelsInt = (blockLabels).astype(np.int32)

# print(blockLabelsInt)
tol = 1e-4
print(ps)
offset = pad-ps//2 # offset for computing GT since we padded images

# vals[vals>10**4]=np.inf
# print(vals[0,:,0])

for _xstart in np.arange(0,w):
    for _ystart in np.arange(0,h):
        # print(xstart,ystart)
        xstart = _xstart + offset
        ystart = _ystart + offset

        res = []
        i = 0
        ref_xy = bfnnf_ref_image[:,xstart:xstart+ps,
                                 ystart:ystart+ps]
        for i in range(blockLabelsInt.shape[0]):
            x,y = blockLabelsInt[i,:]

            x_start = xstart + x
            y_start = ystart + y
            x_end = xstart + x + ps
            y_end = ystart + y + ps
            # print(i,(x_start,x_end),(y_start,y_end))
            if x_start < 0 or y_start < 0:
                # print("continue")
                res.append(np.inf)
                continue
            if x_end > target_image.shape[1]:
                # print("continue")
                res.append(np.inf)
                continue
            if y_end > target_image.shape[2]:
                # print("continue")
                res.append(np.inf)
                continue
            tgt_xy = target_image[:,x_start:x_end,
                                  y_start:y_end]
            ref_xy = torch.Tensor(ref_xy)
            tgt_xy = torch.Tensor(tgt_xy)
            # ref_xy = nnf_ref[:,xstart+pad:xstart+ps+pad,
            #                  ystart+pad:ystart+ps+pad]
            # tgt_xy = nnf_target[:,xstart+x+pad:xstart+x+ps+pad,
            #                     ystart+y+pad:ystart+y+ps+pad]
    
            # ref_xy = ref_xy[0,0,0]
            # tgt_xy = tgt_xy[0,0,0]
            res.append(float(torch.sum(torch.pow( ref_xy - tgt_xy , 2)).item()))
        # res = np.sort(np.array(res))
        val = vals[_xstart,_ystart,:]
        val = np.sort(val)
        loc = locs[_xstart,_ystart,:]
        val[val>10**3] = np.inf
        order = blockLabels[np.argsort(res),:][:len(val)]
        res = np.sort(res)
        res = np.nan_to_num(res,posinf=0.)[:len(val)]
        # print(order[:len(val)],loc)
        # print(res - val)
        # print("-"*10)
        def assert_msg():
            msg = "res" + str(res) + "\n\n"
            msg += "GT order " + str(order) + "\n\n"
            msg += "val" + str(val) + "\n\n"
            msg += "loc" + str(loc) + "\n\n"
            msg += "index: " + str(i)+ "\n\n"
            msg += "(x,y): ({},{})".format(xstart,ystart)+ "\n\n"
            msg += "(x,y): ({},{})".format(_xstart,_ystart)+ "\n\n"
            return msg
        msg = assert_msg()
        if xstart == (h//2) and ystart == (w//2):
            print(msg)
        assert np.mean(np.abs(val-res)) < tol, ("Must be equal. " + msg)
        # print("val, res, val/res",val,res,val / res)
        # print("Expected Output: ",np.array(res))
    
#
# --- Version 2 --
#

# blockLabelsInt = (blockLabels - 1).astype(np.int32)
# xstart,ystart = 4,4
# for xstart in np.arange(h-ps-pad):
#     for ystart in np.arange(w-ps-pad):
#         res = []
#         for i in range(nblocks**2):
#             x,y = blockLabelsInt[i,:]
#             ref_xy = nnf_ref[:,xstart+pad:xstart+ps+pad,
#                              ystart+pad:ystart+ps+pad]
#             tgt_xy = nnf_target[:,xstart+x+pad:xstart+x+ps+pad,
#                                 ystart+y+pad:ystart+y+ps+pad]
#             res.append(float(torch.sum(torch.pow( ref_xy - tgt_xy , 2)).item()))
        # res = np.array(res)
        # print(vals[xstart+1,ystart+1,:] / res)
        # print(np.sqrt(vals[xstart+1,ystart+1,:]) / res)
        # print("Expected Output: ",np.array(res))
        
# print("nnf vals!")
# for i in range(3):
#     print(nnf_vals[0,0,1,i,:].astype(np.float32))

