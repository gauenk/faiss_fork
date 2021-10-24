/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuDistanceDataType.h>

namespace faiss {
namespace gpu {


/// Arguments to brute-force GPU k-nearest neighbor searching
struct GpuKmBurstParams {
    GpuKmBurstParams()
            : metric(faiss::MetricType::METRIC_L2),
              metricArg(0),
              k(0),
              t(0),
              h(0),
              w(0),
              c(0),
              ps(0),
              nsearch(0),
	      kmeansK(0),
              std(0),
              burst(nullptr),
              dType(DistanceDataType::F32),
	      init_blocks(nullptr),
	      search_ranges(nullptr),
              outDistances(nullptr),
	      outIndices(nullptr),
              outIndicesType(IndicesDataType::I64),
              ignoreOutDistances(false) {}


    //
    // Search parameters
    //

    /// Search parameter: distance metric
    faiss::MetricType metric;

    /// Search parameter: distance metric argument (if applicable)
    /// For metric == METRIC_Lp, this is the p-value
    float metricArg;

    /// Search parameter: return k nearest neighbors
    /// If the value provided is -1, then we report all pairwise distances
    /// without top-k filtering
    int k;

    /// image dims
    int t; // nframes
    int h; // height
    int w; // width
    int c; // color
    int ps; // patchsize radius on one direction
    int nsearch; // number of pairs [drow, dcol] for each pixel
    float std;
    int kmeansK; // number of clusters in kmeans
    DistanceDataType dType;// data type of pixels

    //
    // Vectors being queried
    //

    /// If vectorsRowMajor is true, this is
    /// numVectors x dims, with dims innermost; otherwise,
    /// dims x numVectors, with numVectors innermost
    const void* burst;

    /// the initial choice of blocks from the search_ranges at start
    int* init_blocks;

    /// trajectory used to indexing inside of cuda kerenl.
    int* search_ranges;

    //
    // Output results
    //

    /// A region of memory size numQueries x k, with k
    /// innermost (row major) if k > 0, or if k == -1, a region of memory of
    /// size numQueries x numVectors
    float* outDistances;

    /// Do we only care about the indices reported, rather than the output
    /// distances? Not used if k == -1 (all pairwise distances)
    bool ignoreOutDistances;

    /// A region of memory size numQueries x k, with k
    /// innermost (row major). Not used if k == -1 (all pairwise distances)
    IndicesDataType outIndicesType;
    void* outIndices;
};

/// A wrapper for gpu/impl/Distance.cuh to expose direct brute-force k-nearest
/// neighbor searches on an externally-provided region of memory (e.g., from a
/// pytorch tensor).
/// The data (vectors, queries, outDistances, outIndices) can be resident on the
/// GPU or the CPU, but all calculations are performed on the GPU. If the result
/// buffers are on the CPU, results will be copied back when done.
///
/// All GPU computation is performed on the current CUDA device, and ordered
/// with respect to resources->getDefaultStreamCurrentDevice().
///
/// For each vector in `queries`, searches all of `vectors` to find its k
/// nearest neighbors with respect to the given metric

void bfKmBurst(GpuResourcesProvider* resources,
	       const GpuKmBurstParams& args);

}
}

