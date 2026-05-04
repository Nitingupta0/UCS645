#include "core/workload_splitter.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>

namespace gml {

WorkloadSplitter::WorkloadSplitter(int64_t degree_threshold,
                                    int64_t gpu_vertex_limit)
    : degree_threshold_(degree_threshold),
      gpu_vertex_limit_(gpu_vertex_limit) {}

WorkloadPartition WorkloadSplitter::split(const CSRGraph& g) const {
    WorkloadPartition part;

    for (int64_t v = 0; v < g.num_vertices; ++v) {
        if (g.degree(v) > degree_threshold_) {
            part.cpu_vertices.push_back(v);
        } else {
            part.gpu_vertices.push_back(v);
        }
    }


    if (static_cast<int64_t>(part.gpu_vertices.size()) > gpu_vertex_limit_) {
        int64_t spill = (int64_t)part.gpu_vertices.size() - gpu_vertex_limit_;
        for (int64_t i = 0; i < spill; ++i) {
            part.cpu_vertices.push_back(part.gpu_vertices.back());
            part.gpu_vertices.pop_back();
        }
    }

    part.cpu_fraction = (double)part.cpu_vertices.size() / g.num_vertices;

    std::cout << "[WorkloadSplitter] CPU vertices: " << part.cpu_vertices.size()
              << "  GPU vertices: " << part.gpu_vertices.size()
              << "  (threshold=" << degree_threshold_ << ")\n";

    return part;
}

WorkloadPartition WorkloadSplitter::split_adaptive(const CSRGraph& g) const {

    std::mt19937_64 rng(12345ULL);
    int64_t sample_n = std::max((int64_t)100, g.num_vertices / 100);

    std::vector<int64_t> degrees(sample_n);
    std::uniform_int_distribution<int64_t> dist(0, g.num_vertices - 1);
    for (int64_t i = 0; i < sample_n; ++i)
        degrees[i] = g.degree(dist(rng));

    std::sort(degrees.begin(), degrees.end());
    int64_t median_degree = degrees[sample_n / 2];


    int64_t adaptive_threshold = std::max((int64_t)4, median_degree * 2);

    std::cout << "[WorkloadSplitter] Adaptive threshold: " << adaptive_threshold
              << " (median degree=" << median_degree << ")\n";

    WorkloadSplitter adaptive_splitter(adaptive_threshold, gpu_vertex_limit_);
    return adaptive_splitter.split(g);
}

}
