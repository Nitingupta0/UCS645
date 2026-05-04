#pragma once
#include "i_classifier.h"
#include <memory>

namespace gml {





struct TreeNode {
    bool  is_leaf       = false;
    int   feature_idx   = -1;
    float threshold     = 0.0f;
    int   prediction    = -1;
    int   depth         = 0;
    float gini_gain     = 0.0f;
    int   n_samples     = 0;

    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

class DecisionTree : public IClassifier {
public:
    explicit DecisionTree(int max_depth        = 10,
                           int min_samples_leaf = 5,
                           int num_features     = -1,
                           uint64_t seed        = 42)
        : max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          num_features_(num_features),
          seed_(seed) {}

    void fit(const Dataset& train) override;
    std::vector<int>   predict(const FeatureMatrix& X) const override;
    std::vector<float> predict_proba(const FeatureMatrix& X,
                                      int class_idx) const override;
    std::string name() const override { return "DecisionTree"; }

    const TreeNode* root() const { return root_.get(); }
    int num_classes() const { return num_classes_; }


    void accumulate_importance(const TreeNode* node,
                               std::vector<float>& importances,
                               int total_samples) const;

private:
    int      max_depth_;
    int      min_samples_leaf_;
    int      num_features_;
    uint64_t seed_;
    int      num_classes_ = 0;

    std::unique_ptr<TreeNode> root_;

    std::unique_ptr<TreeNode> build_node(
        const FeatureMatrix& X,
        const std::vector<int>& labels,
        const std::vector<int>& sample_idx,
        int depth,
        uint64_t& rng) const;


    std::tuple<int,float,float> best_split(
        const FeatureMatrix& X,
        const std::vector<int>& labels,
        const std::vector<int>& sample_idx,
        const std::vector<int>& feature_subset,
        uint64_t& rng) const;

    float gini_impurity(const std::vector<int>& labels,
                         const std::vector<int>& indices) const;

    int majority_class(const std::vector<int>& labels,
                        const std::vector<int>& indices) const;

    int predict_one(const TreeNode* node, const float* row) const;
};

}
