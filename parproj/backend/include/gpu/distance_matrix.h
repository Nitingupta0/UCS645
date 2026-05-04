#pragma once
#include "i_gpu_processor.h"
#include <vector>

namespace gml {







class DistanceMatrixCUDA {
public:



    std::vector<float> compute(const FeatureMatrix& fm,
                                const std::vector<int64_t>& indices = {});
};

}
