#pragma once
#include "features/i_feature_extractor.h"

namespace gml {




class IGpuProcessor {
public:
    virtual ~IGpuProcessor() = default;


    virtual void process(FeatureMatrix& fm) = 0;


    static bool cuda_available();


    static size_t available_vram();
};

}
