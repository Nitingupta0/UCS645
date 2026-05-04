#pragma once
#include "i_feature_extractor.h"

namespace gml {



class PageRank : public IFeatureExtractor {
public:
    explicit PageRank(double damping   = 0.85,
                       double tolerance = 1e-6,
                       int    max_iter  = 100)
        : damping_(damping), tolerance_(tolerance), max_iter_(max_iter) {}

    FeatureMatrix extract(const CSRGraph& g) override;
    FeatureMatrix extract(const CSRGraph& g,
                           const std::vector<int64_t>& vertices) override;
    std::string name() const override { return "PageRank"; }

private:
    double damping_;
    double tolerance_;
    int    max_iter_;
};

}
