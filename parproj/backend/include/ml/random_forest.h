#pragma once
#include "i_classifier.h"
#include "decision_tree.h"
#include <vector>
#include <memory>

namespace gml {













class ParallelRandomForest : public IClassifier {
public:
    explicit ParallelRandomForest(int    n_trees          = 100,
                                   int    max_depth         = 10,
                                   int    min_samples_leaf  = 5,
                                   double bootstrap_ratio   = 0.8,
                                   int    num_features      = -1,
                                   uint64_t seed            = 42)
        : n_trees_(n_trees),
          max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          bootstrap_ratio_(bootstrap_ratio),
          num_features_(num_features),
          seed_(seed) {}

    void fit(const Dataset& train) override;
    std::vector<int>   predict(const FeatureMatrix& X) const override;
    std::vector<float> predict_proba(const FeatureMatrix& X,
                                      int class_idx) const override;
    std::string name() const override { return "ParallelRandomForest"; }


    std::vector<float> feature_importances() const;

private:
    int      n_trees_;
    int      max_depth_;
    int      min_samples_leaf_;
    double   bootstrap_ratio_;
    int      num_features_;
    uint64_t seed_;
    int      num_classes_  = 0;
    int      num_features_total_ = 0;

    std::vector<std::unique_ptr<DecisionTree>> trees_;


    std::vector<int> bootstrap_sample(int n, uint64_t& rng) const;
};

}
