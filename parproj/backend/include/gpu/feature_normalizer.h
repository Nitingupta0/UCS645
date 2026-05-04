#pragma once
#include "i_gpu_processor.h"

namespace gml {











enum class NormMode { Z_SCORE, MIN_MAX };

class FeatureNormalizerCUDA : public IGpuProcessor {
public:
    explicit FeatureNormalizerCUDA(NormMode mode = NormMode::Z_SCORE,
                                    float eps      = 1e-8f)
        : mode_(mode), eps_(eps) {}

    void process(FeatureMatrix& fm) override;

private:
    NormMode mode_;
    float    eps_;
};

}
