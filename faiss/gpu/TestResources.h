#pragma once


namespace faiss {
  namespace gpu {

    enum KM_TEST_TYPE {
	KmOther = -1,
	PairwiseDistanceCase = 0,
	ClusterUpdate = 1,
	CentroidUpdate = 2,
	Ranges2Blocks = 3,
	KMeansCase = 4,
	ComputeModeCase = 5,
	KmBurstAveCase = 6,
	KmBurstL2NormCase = 7,
	KmBurstTopKCase = 8
    };

  }
}
