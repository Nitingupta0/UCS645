#include "features/clustering_coefficient.h"
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {



int64_t ClusteringCoefficient::count_triangles(const CSRGraph& g,
                                                 int64_t v) const {
    int64_t triangles = 0;
    auto nb_begin = g.neighbors_begin(v);
    auto nb_end   = g.neighbors_end(v);

    for (auto it = nb_begin; it != nb_end; ++it) {
        int64_t u = *it;
        if (u == v) continue;


        auto uit = g.neighbors_begin(u);
        auto uend = g.neighbors_end(u);

        auto vit = nb_begin;
        while (vit != nb_end && uit != uend) {
            if (*vit == *uit) {
                if (*vit != v && *vit != u) ++triangles;
                ++vit; ++uit;
            } else if (*vit < *uit) {
                ++vit;
            } else {
                ++uit;
            }
        }
    }
    return triangles / 2;
}

FeatureMatrix ClusteringCoefficient::extract(const CSRGraph& g) {
    std::vector<int64_t> all(g.num_vertices);
    std::iota(all.begin(), all.end(), 0);
    return extract(g, all);
}

FeatureMatrix ClusteringCoefficient::extract(const CSRGraph& g,
                                              const std::vector<int64_t>& vertices) {
    int64_t n = static_cast<int64_t>(vertices.size());

    FeatureMatrix fm;
    fm.num_vertices = n;
    fm.num_features = 1;
    fm.names        = {"clustering_coefficient"};
    fm.data.resize(n, 0.0f);


    #pragma omp parallel for schedule(dynamic, 8) default(none) \
        shared(g, vertices, fm, n)
    for (int64_t i = 0; i < n; ++i) {
        int64_t v  = vertices[i];
        int64_t dv = g.degree(v);

        if (dv < 2) {
            fm.data[i] = 0.0f;
            continue;
        }

        int64_t tri   = count_triangles(g, v);
        double  possible = (double)dv * (dv - 1) / 2.0;
        fm.data[i] = static_cast<float>(tri / possible);
    }

    return fm;
}

}
