#pragma once
#include "i_feature_extractor.h"

namespace gml {



class DegreeCentrality : public IFeatureExtractor {
public:
    explicit DegreeCentrality(bool normalise = true)
        : normalise_(normalise) {}

    FeatureMatrix extract(const CSRGraph& g) override;
    FeatureMatrix extract(const CSRGraph& g,
                           const std::vector<int64_t>& vertices) override;
    std::string name() const override { return "DegreeCentrality"; }

private:
    bool normalise_;
};

}
