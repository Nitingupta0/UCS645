#include "features/degree_centrality.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {

FeatureMatrix DegreeCentrality::extract(const CSRGraph& g) {
    std::vector<int64_t> all(g.num_vertices);
    for (int64_t v = 0; v < g.num_vertices; ++v) all[v] = v;
    return extract(g, all);
}

FeatureMatrix DegreeCentrality::extract(const CSRGraph& g,
                                         const std::vector<int64_t>& vertices) {
    int64_t n     = static_cast<int64_t>(vertices.size());
    int     F     = g.directed ? 3 : 1;
    float   denom = normalise_ && g.num_vertices > 1
                    ? static_cast<float>(g.num_vertices - 1) : 1.0f;

    FeatureMatrix fm;
    fm.num_vertices = n;
    fm.num_features = F;
    fm.data.resize(n * F, 0.0f);

    if (g.directed) {
        fm.names = {"in_degree", "out_degree", "total_degree"};

        std::vector<float> in_deg(g.num_vertices, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int64_t v = 0; v < (int64_t)g.num_vertices; ++v)
            for (int64_t e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e)
                #pragma omp atomic
                in_deg[g.col_idx[e]] += 1.0f;

        #pragma omp parallel for schedule(guided) default(none) \
            shared(fm, g, vertices, in_deg, denom, F, n)
        for (int64_t i = 0; i < n; ++i) {
            int64_t v  = vertices[i];
            float   od = static_cast<float>(g.degree(v));
            float   id = in_deg[v];
            fm.at(i, 0) = id / denom;
            fm.at(i, 1) = od / denom;
            fm.at(i, 2) = (id + od) / (2.0f * denom);
        }
    } else {
        fm.names = {"degree_centrality"};

        #pragma omp parallel for schedule(guided) default(none) \
            shared(fm, g, vertices, denom, n)
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = vertices[i];
            fm.at(i, 0) = static_cast<float>(g.degree(v)) / denom;
        }
    }

    return fm;
}

}
