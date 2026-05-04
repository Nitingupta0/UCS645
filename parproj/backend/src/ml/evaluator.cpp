#include "ml/evaluator.h"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>

namespace gml {




std::pair<Dataset, Dataset> Dataset::train_test_split(double test_ratio,
                                                       uint64_t seed) const {
    int n = static_cast<int>(features.num_vertices);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    int n_test  = static_cast<int>(n * test_ratio);
    int n_train = n - n_test;

    auto build_subset = [&](int start, int count) {
        Dataset d;
        d.num_classes = num_classes;
        d.features.num_vertices = count;
        d.features.num_features = features.num_features;
        d.features.names        = features.names;
        d.features.data.resize(count * features.num_features);
        d.labels.resize(count);
        for (int i = 0; i < count; ++i) {
            int src = perm[start + i];
            d.labels[i] = labels[src];
            for (int64_t f = 0; f < features.num_features; ++f)
                d.features.data[i * features.num_features + f]
                    = features.at(src, f);
        }
        return d;
    };

    Dataset train_ds = build_subset(0,       n_train);
    Dataset test_ds  = build_subset(n_train, n_test);
    return {std::move(train_ds), std::move(test_ds)};
}

std::string ClassificationReport::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "\n══════════════════ Classification Report ══════════════════\n";
    oss << std::setw(8) << "Class"
        << std::setw(12) << "Precision"
        << std::setw(12) << "Recall"
        << std::setw(12) << "F1-Score"
        << std::setw(10) << "Support" << "\n";
    oss << std::string(54, '-') << "\n";

    for (int c = 0; c < (int)per_class.size(); ++c) {
        const auto& pc = per_class[c];
        oss << std::setw(8)  << c
            << std::setw(12) << pc.precision
            << std::setw(12) << pc.recall
            << std::setw(12) << pc.f1
            << std::setw(10) << pc.support << "\n";
    }

    oss << std::string(54, '-') << "\n";
    oss << std::setw(8) << "Macro"
        << std::setw(12) << macro_precision
        << std::setw(12) << macro_recall
        << std::setw(12) << macro_f1 << "\n\n";
    oss << "  Accuracy: " << accuracy << "\n";
    oss << "═══════════════════════════════════════════════════════════\n";
    return oss.str();
}

ClassificationReport Evaluator::evaluate(const std::vector<int>& y_true,
                                           const std::vector<int>& y_pred) const {
    ClassificationReport report;
    int n = static_cast<int>(y_true.size());


    report.confusion_matrix.assign(num_classes_,
                                    std::vector<int>(num_classes_, 0));
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        report.confusion_matrix[y_true[i]][y_pred[i]]++;
        if (y_true[i] == y_pred[i]) ++correct;
    }
    report.accuracy = (double)correct / n;


    report.per_class.resize(num_classes_);
    for (int c = 0; c < num_classes_; ++c) {
        int tp = report.confusion_matrix[c][c];
        int fp = 0, fn = 0;
        for (int j = 0; j < num_classes_; ++j) {
            if (j != c) fp += report.confusion_matrix[j][c];
            if (j != c) fn += report.confusion_matrix[c][j];
        }
        auto& pc    = report.per_class[c];
        pc.support   = tp + fn;
        pc.precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        pc.recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        pc.f1        = (pc.precision + pc.recall > 0)
                       ? 2 * pc.precision * pc.recall / (pc.precision + pc.recall)
                       : 0.0;
    }


    double sum_p = 0, sum_r = 0, sum_f = 0;
    for (auto& pc : report.per_class) { sum_p += pc.precision; sum_r += pc.recall; sum_f += pc.f1; }
    report.macro_precision = sum_p / num_classes_;
    report.macro_recall    = sum_r / num_classes_;
    report.macro_f1        = sum_f / num_classes_;

    return report;
}

ClassificationReport Evaluator::cross_validate(IClassifier& clf,
                                                 const Dataset& dataset,
                                                 int k_folds,
                                                 uint64_t seed) const {
    int n = static_cast<int>(dataset.features.num_vertices);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<int> all_pred(n, -1);

    for (int fold = 0; fold < k_folds; ++fold) {
        int test_start = fold * n / k_folds;
        int test_end   = (fold + 1) * n / k_folds;


        std::vector<int> train_idx, test_idx;
        for (int i = 0; i < n; ++i) {
            if (i >= test_start && i < test_end) test_idx.push_back(perm[i]);
            else                                  train_idx.push_back(perm[i]);
        }


        auto make_subset = [&](const std::vector<int>& idx) {
            Dataset d;
            d.num_classes = dataset.num_classes;
            d.features.num_vertices = idx.size();
            d.features.num_features = dataset.features.num_features;
            d.features.names        = dataset.features.names;
            d.features.data.resize(idx.size() * dataset.features.num_features);
            d.labels.resize(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) {
                d.labels[i] = dataset.labels[idx[i]];
                for (int64_t f = 0; f < dataset.features.num_features; ++f)
                    d.features.data[i * dataset.features.num_features + f]
                        = dataset.features.at(idx[i], f);
            }
            return d;
        };

        Dataset train = make_subset(train_idx);
        Dataset test  = make_subset(test_idx);

        auto t0 = std::chrono::high_resolution_clock::now();
        clf.fit(train);
        auto preds = clf.predict(test.features);
        auto t1 = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_idx.size(); ++i)
            all_pred[test_idx[i]] = preds[i];

        double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
        std::cout << "[CV] Fold " << fold+1 << "/" << k_folds << " in " << ms << " ms\n";
    }

    return evaluate(dataset.labels, all_pred);
}

void Evaluator::print_timing(const std::string& phase, double elapsed_ms) {
    std::cout << std::fixed << std::setprecision(2)
              << "[Timing] " << phase << ": " << elapsed_ms << " ms\n";
}

}
