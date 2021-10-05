import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor


def KMeans(x, K=10, Niter=10, verbose=False, randDist=0.):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    if K > x.shape[1]: K = x.shape[1]
    
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64


    start = time.time()
    B, N, D = x.shape  # num batches, Number of samples, dimension of the ambient space

    c = x[:, :K, :].clone()  # Simplistic initialization for the centroids

    if randDist > 0: x = torch.normal(x,randDist)

    x_i = LazyTensor(x.view(B, N, 1, D))  # (B, N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (B, 1, K, D) centroids

    # K-means loop:
    # - x  is the (B, N, D) point cloud,
    # - cl is the (B, N,) vector of class labels
    # - c  is the (B, K, D) cloud of cluster centroids
    # - Ncl is the (B, K, D) cloud of number of elems per cluster centroid
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, N, K) symbolic squared distances
        cl = D_ij.argmin(dim=2).long().view(B,-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(1, cl[:, :, None].repeat(1, 1, D), x)

        # Divide by the number of points per cluster:
        # Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        max_value = K
        Ncl = batched_bincount(cl, 1, max_value).type_as(c).view(B, K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c, Ncl



def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target
