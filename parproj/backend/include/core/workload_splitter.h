#pragma once
#include "graph.h"
#include <vector>
#include <cstdint>

namespace gml {










struct WorkloadPartition {
    std::vector<int64_t> cpu_vertices;
    std::vector<int64_t> gpu_vertices;
    double cpu_fraction = 0.0;
};

class WorkloadSplitter {
public:


    explicit WorkloadSplitter(int64_t degree_threshold = 32,
                               int64_t gpu_vertex_limit = 1'000'000);

    WorkloadPartition split(const CSRGraph& g) const;


    WorkloadPartition split_adaptive(const CSRGraph& g) const;

private:
    int64_t degree_threshold_;
    int64_t gpu_vertex_limit_;
};

}
