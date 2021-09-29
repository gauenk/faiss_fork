
/// convert block indices to labels



#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/StaticUtils.h>

namespace faiss {
namespace gpu {


  template <int RowTileSize, int ColTileSize, int KTileSize>
  __global__ void blockIndices2Labels(
	   Tensor<int, 3, true> inIndices,
	   Tensor<int, 5, true> outLabels,
	   Tensor<int, 3, true> blocks){
    
    // row col access
    int k = threadIdx.x;
    int rowStart = RowTileSize * blockIdx.x;
    int colStart = ColTileSize * blockIdx.y;
    int nframes = blocks.getSize(0);
    bool lastRow = (rowStart+RowTileSize - 1) >= inIndices.getSize(0);
    bool lastCol = (colStart+ColTileSize - 1) >= inIndices.getSize(1);
    bool lastAccess = lastRow && lastCol;

    // run function if legal
    if (lastAccess){
      int rowNumel = inIndices.getSize(0) - rowStart;
      int colNumel = inIndices.getSize(1) - colStart;

      for (int frame = 0; frame < nframes; ++frame){
	for (int row = 0; row < rowNumel; ++row){
	  for (int col = 0; col < colNumel; ++col){
	    int rowIdx = rowStart + row;
	    int colIdx = colStart + col;
	    int blIndex = inIndices[rowIdx][colIdx][k];
	    outLabels[frame][rowIdx][colIdx][k][0] = blocks[frame][blIndex][0];
	    outLabels[frame][rowIdx][colIdx][k][1] = blocks[frame][blIndex][1];
	  }
	}
      }      

    } else {
      for (int frame = 0; frame < nframes; ++frame){
#pragma unroll
	for (int row = 0; row < RowTileSize; ++row){
#pragma unroll
	  for (int col = 0; col < ColTileSize; ++col){
	    int rowIdx = rowStart + row;
	    int colIdx = colStart + col;
	    int blIndex = inIndices[rowIdx][colIdx][k];
	    outLabels[frame][rowIdx][colIdx][k][0] = blocks[frame][blIndex][0];
	    outLabels[frame][rowIdx][colIdx][k][1] = blocks[frame][blIndex][1];
	  }
	}
      }      

    }

  }

  void runBlockIndices2Labels(
          Tensor<int, 3, true>& inIndices,
	  Tensor<int, 5, true>& outLabels,
	  Tensor<int, 3, true>& blocks,
	  cudaStream_t stream){

    // batching per thread
    constexpr int RowTileSize = 2;
    constexpr int ColTileSize = 2;
    constexpr int KTileSize = 1;

    // error checking
    FAISS_ASSERT(inIndices.getSize(0) == outLabels.getSize(1)); // height
    FAISS_ASSERT(inIndices.getSize(1) == outLabels.getSize(2)); // width
    FAISS_ASSERT(inIndices.getSize(2) == outLabels.getSize(3)); // k
    FAISS_ASSERT(blocks.getSize(0) == outLabels.getSize(0)); // nframes

    // get num of threads
    int maxThreads = getMaxThreadsCurrentDevice();
    FAISS_ASSERT(maxThreads > outLabels.getSize(3)); // k < maxNumThreads
    int nThreads = std::min(outLabels.getSize(3),maxThreads);
    
    // get tile sizes
    int heightTile = utils::divUp(inIndices.getSize(0),RowTileSize);
    int widthTile = utils::divUp(inIndices.getSize(1),ColTileSize);
    auto grid = dim3(heightTile,widthTile);
    auto dim = dim3(nThreads);
    
    blockIndices2Labels<RowTileSize,ColTileSize,KTileSize>
      <<<grid, dim, 0, stream>>>(inIndices,outLabels,blocks);

  }


}
}