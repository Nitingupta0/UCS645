#include "ml/random_forest.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {

std::vector<int> ParallelRandomForest::bootstrap_sample(int n,
                                                          uint64_t& rng_state) const {
    int sample_n = static_cast<int>(n * bootstrap_ratio_);
    std::mt19937_64 r(rng_state++);
    std::uniform_int_distribution<int> dist(0, n - 1);
    std::vector<int> idx(sample_n);
    for (auto& i : idx) i = dist(r);
    return idx;
}

void ParallelRandomForest::fit(const Dataset& train) {
    num_classes_       = train.num_classes;
    num_features_total_ = static_cast<int>(train.features.num_features);
    int n              = static_cast<int>(train.features.num_vertices);

    trees_.resize(n_trees_);

    auto t0 = std::chrono::high_resolution_clock::now();


    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(train, n)
    for (int t = 0; t < n_trees_; ++t) {

        uint64_t tree_seed = seed_ ^ ((uint64_t)t * 6364136223846793005ULL + 1);


        uint64_t rng = tree_seed;
        std::vector<int> sample_idx = bootstrap_sample(n, rng);


        Dataset sub;
        sub.num_classes = num_classes_;
        sub.features.num_vertices = static_cast<int64_t>(sample_idx.size());
        sub.features.num_features = train.features.num_features;
        sub.features.names        = train.features.names;
        sub.features.data.resize(sample_idx.size() * train.features.num_features);
        sub.labels.resize(sample_idx.size());

        for (size_t i = 0; i < sample_idx.size(); ++i) {
            int src = sample_idx[i];
            sub.labels[i] = train.labels[src];
            for (int64_t f = 0; f < train.features.num_features; ++f)
                sub.features.data[i * train.features.num_features + f]
                    = train.features.at(src, f);
        }

        trees_[t] = std::make_unique<DecisionTree>(
            max_depth_, min_samples_leaf_, num_features_, tree_seed);
        trees_[t]->fit(sub);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    std::cout << "[RandomForest] Trained " << n_trees_ << " trees in "
              << ms << " ms  (" << ms / n_trees_ << " ms/tree avg)\n";
}

std::vector<int> ParallelRandomForest::predict(const FeatureMatrix& X) const {
    int64_t n = X.num_vertices;

    std::vector<std::vector<int>> votes(n, std::vector<int>(num_classes_, 0));


    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(X, votes, n)
    for (int t = 0; t < n_trees_; ++t) {
        auto preds = trees_[t]->predict(X);
        for (int64_t v = 0; v < n; ++v) {
            #pragma omp atomic
            votes[v][preds[v]]++;
        }
    }

    std::vector<int> result(n);
    for (int64_t v = 0; v < n; ++v)
        result[v] = static_cast<int>(
            std::max_element(votes[v].begin(), votes[v].end()) - votes[v].begin());
    return result;
}

std::vector<float> ParallelRandomForest::predict_proba(const FeatureMatrix& X,
                                                         int class_idx) const {
    int64_t n = X.num_vertices;
    std::vector<float> proba(n, 0.0f);

    for (int t = 0; t < n_trees_; ++t) {
        auto p = trees_[t]->predict_proba(X, class_idx);
        for (int64_t v = 0; v < n; ++v)
            proba[v] += p[v];
    }
    for (auto& p : proba) p /= n_trees_;
    return proba;
}

std::vector<float> ParallelRandomForest::feature_importances() const {


    std::vector<float> imp(num_features_total_, 0.0f);

    for (int t = 0; t < n_trees_; ++t) {
        const auto* root = trees_[t]->root();
        if (!root) continue;
        std::vector<float> tree_imp(num_features_total_, 0.0f);
        trees_[t]->accumulate_importance(root, tree_imp, root->n_samples);
        for (int f = 0; f < num_features_total_; ++f)
            imp[f] += tree_imp[f];
    }


    float total = 0.0f;
    for (float v : imp) total += v;
    if (total > 0.0f) {
        for (float& v : imp) v /= total;
    }
    return imp;
}

}
