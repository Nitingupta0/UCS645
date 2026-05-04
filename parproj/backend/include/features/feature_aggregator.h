#pragma once
#include "i_feature_extractor.h"
#include <vector>
#include <memory>

namespace gml {





class FeatureAggregator {
public:

    void add_extractor(std::unique_ptr<IFeatureExtractor> extractor);



    FeatureMatrix aggregate(const CSRGraph& g) const;


    FeatureMatrix aggregate(const CSRGraph& g,
                             const std::vector<int64_t>& vertices) const;

    int num_extractors() const { return static_cast<int>(extractors_.size()); }

private:
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors_;
};

}
