#include "features/feature_aggregator.h"
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {

void FeatureMatrix::append_columns(const FeatureMatrix& other) {
    if (num_vertices != other.num_vertices)
        throw std::runtime_error("FeatureMatrix::append_columns: vertex count mismatch");

    int64_t new_F = num_features + other.num_features;
    std::vector<float> new_data(num_vertices * new_F);

    for (int64_t v = 0; v < num_vertices; ++v) {
        for (int64_t f = 0; f < num_features; ++f)
            new_data[v * new_F + f] = at(v, f);
        for (int64_t f = 0; f < other.num_features; ++f)
            new_data[v * new_F + num_features + f] = other.at(v, f);
    }

    data         = std::move(new_data);
    num_features = new_F;
    names.insert(names.end(), other.names.begin(), other.names.end());
}

void FeatureAggregator::add_extractor(
    std::unique_ptr<IFeatureExtractor> extractor) {
    extractors_.push_back(std::move(extractor));
}

FeatureMatrix FeatureAggregator::aggregate(const CSRGraph& g) const {
    std::vector<int64_t> all(g.num_vertices);
    for (int64_t v = 0; v < g.num_vertices; ++v) all[v] = v;
    return aggregate(g, all);
}

FeatureMatrix FeatureAggregator::aggregate(
    const CSRGraph& g, const std::vector<int64_t>& vertices) const {

    if (extractors_.empty())
        throw std::runtime_error("FeatureAggregator: no extractors registered");

    int E = static_cast<int>(extractors_.size());
    std::vector<FeatureMatrix> results(E);
    std::vector<double>        timings(E, 0.0);
    std::vector<std::string>   names(E);

    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(g, vertices, results, timings, names, E)
    for (int i = 0; i < E; ++i) {
        auto t0   = std::chrono::high_resolution_clock::now();
        results[i] = extractors_[i]->extract(g, vertices);
        auto t1   = std::chrono::high_resolution_clock::now();
        timings[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        names[i]   = extractors_[i]->name();
    }


    for (int i = 0; i < E; ++i)
        std::cout << "[Aggregator] " << names[i]
                  << " -> " << results[i].num_features
                  << " features in " << timings[i] << " ms\n";


    FeatureMatrix combined = std::move(results[0]);
    for (int i = 1; i < E; ++i)
        combined.append_columns(results[i]);

    std::cout << "[Aggregator] Total features: " << combined.num_features
              << " for " << combined.num_vertices << " vertices\n";
    return combined;
}

}
