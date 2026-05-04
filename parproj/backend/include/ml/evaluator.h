#pragma once
#include "i_classifier.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace gml {




struct ClassificationReport {
    double accuracy       = 0.0;
    double macro_f1       = 0.0;
    double macro_precision= 0.0;
    double macro_recall   = 0.0;

    struct PerClass {
        double precision = 0.0;
        double recall    = 0.0;
        double f1        = 0.0;
        int    support   = 0;
    };
    std::vector<PerClass> per_class;


    std::vector<std::vector<int>> confusion_matrix;

    std::string to_string() const;
};




class Evaluator {
public:
    explicit Evaluator(int num_classes) : num_classes_(num_classes) {}

    ClassificationReport evaluate(const std::vector<int>& y_true,
                                   const std::vector<int>& y_pred) const;


    ClassificationReport cross_validate(IClassifier& clf,
                                         const Dataset& dataset,
                                         int k_folds = 5,
                                         uint64_t seed = 42) const;


    static void print_timing(const std::string& phase, double elapsed_ms);

private:
    int num_classes_;
};

}
