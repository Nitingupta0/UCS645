#include "ml/decision_tree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {




float DecisionTree::gini_impurity(const std::vector<int>& labels,
                                    const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0f;
    std::vector<int> counts(num_classes_, 0);
    for (int i : indices) counts[labels[i]]++;
    float n   = static_cast<float>(indices.size());
    float g   = 1.0f;
    for (int c : counts) g -= (c / n) * (c / n);
    return g;
}

int DecisionTree::majority_class(const std::vector<int>& labels,
                                   const std::vector<int>& indices) const {
    std::vector<int> counts(num_classes_, 0);
    for (int i : indices) counts[labels[i]]++;
    return static_cast<int>(
        std::max_element(counts.begin(), counts.end()) - counts.begin());
}




std::tuple<int,float,float> DecisionTree::best_split(
    const FeatureMatrix& X,
    const std::vector<int>& labels,
    const std::vector<int>& sample_idx,
    const std::vector<int>& feature_subset,
    uint64_t& ) const {

    int    best_f     = -1;
    float  best_thr   = 0.0f;
    float  best_gain  = -1.0f;
    float  parent_g   = gini_impurity(labels, sample_idx);
    int    n          = static_cast<int>(sample_idx.size());



    #pragma omp parallel default(none) \
        shared(X, labels, sample_idx, feature_subset, parent_g, n, \
               best_f, best_thr, best_gain)
    {
        int   local_best_f    = -1;
        float local_best_thr  = 0.0f;
        float local_best_gain = -1.0f;

        #pragma omp for schedule(dynamic)
        for (int fi = 0; fi < (int)feature_subset.size(); ++fi) {
            int f = feature_subset[fi];


            std::vector<int> sorted_idx = sample_idx;
            std::sort(sorted_idx.begin(), sorted_idx.end(),
                      [&](int a, int b){ return X.at(a, f) < X.at(b, f); });


            std::vector<int> left_counts(num_classes_, 0);
            std::vector<int> right_counts(num_classes_, 0);


            for (int k = 0; k < n; ++k)
                right_counts[labels[sorted_idx[k]]]++;

            int left_n = 0;
            int right_n = n;

            for (int k = 0; k < n - 1; ++k) {

                int label_k = labels[sorted_idx[k]];
                left_counts[label_k]++;
                right_counts[label_k]--;
                left_n++;
                right_n--;


                if (X.at(sorted_idx[k], f) == X.at(sorted_idx[k+1], f)) continue;

                float thr = (X.at(sorted_idx[k], f) + X.at(sorted_idx[k+1], f)) / 2.0f;


                float gl = 1.0f;
                for (int c = 0; c < num_classes_; ++c) {
                    float p = static_cast<float>(left_counts[c]) / left_n;
                    gl -= p * p;
                }
                float gr = 1.0f;
                for (int c = 0; c < num_classes_; ++c) {
                    float p = static_cast<float>(right_counts[c]) / right_n;
                    gr -= p * p;
                }

                float gain = parent_g
                           - (gl * left_n + gr * right_n) / n;

                if (gain > local_best_gain) {
                    local_best_gain = gain;
                    local_best_f    = f;
                    local_best_thr  = thr;
                }
            }
        }


        #pragma omp critical
        {
            if (local_best_gain > best_gain) {
                best_gain = local_best_gain;
                best_f    = local_best_f;
                best_thr  = local_best_thr;
            }
        }
    }

    return {best_f, best_thr, best_gain};
}




std::unique_ptr<TreeNode> DecisionTree::build_node(
    const FeatureMatrix& X,
    const std::vector<int>& labels,
    const std::vector<int>& sample_idx,
    int depth,
    uint64_t& rng) const {

    auto node = std::make_unique<TreeNode>();
    node->depth = depth;
    node->n_samples = static_cast<int>(sample_idx.size());


    bool pure = true;
    for (int i : sample_idx) if (labels[i] != labels[sample_idx[0]]) { pure = false; break; }

    if (pure || depth >= max_depth_
             || (int)sample_idx.size() <= min_samples_leaf_) {
        node->is_leaf    = true;
        node->prediction = majority_class(labels, sample_idx);
        return node;
    }


    int F      = static_cast<int>(X.num_features);
    int n_feat = (num_features_ > 0) ? num_features_
                                     : std::max(1, (int)std::sqrt(F));
    n_feat = std::min(n_feat, F);

    std::mt19937_64 r(rng++);
    std::vector<int> all_feats(F);
    std::iota(all_feats.begin(), all_feats.end(), 0);
    std::shuffle(all_feats.begin(), all_feats.end(), r);
    all_feats.resize(n_feat);

    auto [best_f, best_thr, best_gain] = best_split(X, labels, sample_idx,
                                                      all_feats, rng);

    if (best_f < 0 || best_gain <= 0.0f) {
        node->is_leaf    = true;
        node->prediction = majority_class(labels, sample_idx);
        return node;
    }


    std::vector<int> left_idx, right_idx;
    for (int i : sample_idx) {
        if (X.at(i, best_f) <= best_thr) left_idx.push_back(i);
        else                               right_idx.push_back(i);
    }

    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf    = true;
        node->prediction = majority_class(labels, sample_idx);
        return node;
    }

    node->feature_idx = best_f;
    node->threshold   = best_thr;
    node->gini_gain   = best_gain;
    node->left        = build_node(X, labels, left_idx,  depth + 1, rng);
    node->right       = build_node(X, labels, right_idx, depth + 1, rng);
    return node;
}

void DecisionTree::fit(const Dataset& train) {
    num_classes_ = train.num_classes;
    std::vector<int> all_idx(train.features.num_vertices);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    uint64_t rng = seed_;
    root_ = build_node(train.features, train.labels, all_idx, 0, rng);
}

int DecisionTree::predict_one(const TreeNode* node, const float* row) const {
    while (!node->is_leaf) {
        node = (row[node->feature_idx] <= node->threshold)
               ? node->left.get() : node->right.get();
    }
    return node->prediction;
}

std::vector<int> DecisionTree::predict(const FeatureMatrix& X) const {
    int64_t n = X.num_vertices;
    std::vector<int> preds(n);
    #pragma omp parallel for schedule(static) default(none) \
        shared(X, preds, n)
    for (int64_t v = 0; v < n; ++v)
        preds[v] = predict_one(root_.get(), X.data.data() + v * X.num_features);
    return preds;
}

std::vector<float> DecisionTree::predict_proba(const FeatureMatrix& X,
                                                 int class_idx) const {
    auto preds = predict(X);
    std::vector<float> proba(X.num_vertices);
    for (int64_t v = 0; v < X.num_vertices; ++v)
        proba[v] = (preds[v] == class_idx) ? 1.0f : 0.0f;
    return proba;
}




void DecisionTree::accumulate_importance(const TreeNode* node,
                                          std::vector<float>& importances,
                                          int total_samples) const {
    if (!node || node->is_leaf) return;


    float weight = static_cast<float>(node->n_samples) / total_samples;
    importances[node->feature_idx] += weight * node->gini_gain;

    accumulate_importance(node->left.get(),  importances, total_samples);
    accumulate_importance(node->right.get(), importances, total_samples);
}

}
