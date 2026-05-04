#pragma once
#include "i_feature_extractor.h"

namespace gml {





class BetweennessCentrality : public IFeatureExtractor {
public:

    explicit BetweennessCentrality(int64_t num_samples = 0,
                                    uint64_t seed = 42,
                                    bool normalise = true)
        : num_samples_(num_samples), seed_(seed), normalise_(normalise) {}

    FeatureMatrix extract(const CSRGraph& g) override;
    FeatureMatrix extract(const CSRGraph& g,
                           const std::vector<int64_t>& vertices) override;
    std::string name() const override { return "BetweennessCentrality"; }

private:
    int64_t  num_samples_;
    uint64_t seed_;
    bool     normalise_;


    void brandes_bfs(const CSRGraph& g, int64_t source,
                     std::vector<double>& delta,
                     std::vector<double>& bc_local) const;
};

}
