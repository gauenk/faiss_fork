"""
Create meshgrid for BurstNnnf

"""

import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
from pyutils.align_utils.blocks import mesh_block_ranges

def create_meshgrid_blockLabels(blockLabels,select,nframes):
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

def getBlockLabels(blockLabels,nblocks,dtype,device,is_tensor,t=None):
    blockLabels = getBlockLabelsNumpy(blockLabels,nblocks,dtype,t)
    blockLabels = flip_array_like(blockLabels,1)
    if not(torch.is_tensor(blockLabels)) and is_tensor:
        if device != 'cpu':
            blockLabels = torch.cuda.IntTensor(blockLabels,device=device)
        else:
            blockLabels = torch.IntTensor(blockLabels,device=device)
    if torch.is_tensor(blockLabels):
        blockLabels_ptr,_ = torch2swig(blockLabels)
    else:
        blockLabels_ptr = faiss.swig_ptr(blockLabels)
    return blockLabels,blockLabels_ptr

def getBlockLabelsNumpy(blockLabels,nblocks,dtype,t=None):
    if blockLabels is None:
        blockLabels = np.zeros((nblocks**2,2))
        blockLabelsTmp = np.arange(nblocks**2).reshape(nblocks,nblocks)
        for i in range(nblocks**2):
            x,y = np.where(blockLabelsTmp == i)
            blockLabels[i,0] = x
            blockLabels[i,1] = y
        blockLabels -= nblocks//2
        if not(t is None):
            blockLabels = create_meshgrid_blockLabels(blockLabels,t)
            # blockLabels = repeat(blockLabels,'b two -> t b two',t=t)
    if isinstance(blockLabels,np.ndarray):
        return blockLabels.astype(np.int32)
    elif torch.is_tensor(blockLabels):
        return blockLabels.type(torch.int32)
    else:
        return blockLabels

def foo():
    # -- if too many frames --
    if nframes > 7:
        no_search = list(np.random.permutation(nframes))
        no_search.remove(nframes//2)
        no_search = no_search[:nframes-5]
        for ns in no_search: to_mesh[ns] = [[nblocks2//2]]
        mesh_block_ranges(blocks_t,brange,curr_blocks,device=None)

