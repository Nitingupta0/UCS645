#pragma once
#include "i_feature_extractor.h"

namespace gml {




class ClusteringCoefficient : public IFeatureExtractor {
public:
    FeatureMatrix extract(const CSRGraph& g) override;
    FeatureMatrix extract(const CSRGraph& g,
                           const std::vector<int64_t>& vertices) override;
    std::string name() const override { return "ClusteringCoefficient"; }

private:

    int64_t count_triangles(const CSRGraph& g, int64_t v) const;
};

}
