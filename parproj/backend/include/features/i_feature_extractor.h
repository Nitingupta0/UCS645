#pragma once
#include "core/graph.h"
#include <vector>
#include <string>

namespace gml {




struct FeatureMatrix {
    int64_t num_vertices = 0;
    int64_t num_features = 0;
    std::vector<float> data;
    std::vector<std::string> names;

    float& at(int64_t vertex, int64_t feature) {
        return data[vertex * num_features + feature];
    }
    float at(int64_t vertex, int64_t feature) const {
        return data[vertex * num_features + feature];
    }


    void append_columns(const FeatureMatrix& other);
};




class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;


    virtual FeatureMatrix extract(const CSRGraph& g) = 0;


    virtual FeatureMatrix extract(const CSRGraph& g,
                                   const std::vector<int64_t>& vertices) = 0;


    virtual std::string name() const = 0;
};

}
