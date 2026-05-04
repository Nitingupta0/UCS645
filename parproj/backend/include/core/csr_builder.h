#pragma once
#include "graph.h"
#include <vector>
#include <utility>

namespace gml {







class CSRBuilder {
public:
    explicit CSRBuilder(bool directed = false) : directed_(directed) {}


    void add_edge(int64_t src, int64_t dst, float weight = 1.0f);


    void add_edges(const std::vector<std::tuple<int64_t,int64_t,float>>& edges);


    CSRGraph build();


    void reset();

private:
    bool directed_;

    std::vector<std::tuple<int64_t,int64_t,float>> edge_list_;
    int64_t max_vertex_ = -1;
};

}
