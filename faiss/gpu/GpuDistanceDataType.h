#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace gpu {

  class GpuResourcesProvider;


  // Scalar type of the vector data
  enum class DistanceDataType {
    F32 = 1,
      F16,
      };

  // Scalar type of the indices data
  enum class IndicesDataType {
    I64 = 1,
      I32,
      };

}
}
