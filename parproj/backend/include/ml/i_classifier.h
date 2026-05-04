#pragma once
#include "features/i_feature_extractor.h"
#include <vector>
#include <string>

namespace gml {




struct Dataset {
    FeatureMatrix features;
    std::vector<int> labels;
    int num_classes = 0;


    std::pair<Dataset, Dataset> train_test_split(double test_ratio = 0.2,
                                                  uint64_t seed     = 42) const;
};




class IClassifier {
public:
    virtual ~IClassifier() = default;
    virtual void fit(const Dataset& train)                          = 0;
    virtual std::vector<int> predict(const FeatureMatrix& X) const = 0;
    virtual std::vector<float> predict_proba(const FeatureMatrix& X,
                                              int class_idx) const  = 0;
    virtual std::string name() const                                = 0;
};

}
