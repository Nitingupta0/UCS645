#include "features/pagerank.h"
#include <cmath>
#include <numeric>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {

FeatureMatrix PageRank::extract(const CSRGraph& g) {
    std::vector<int64_t> all(g.num_vertices);
    std::iota(all.begin(), all.end(), 0);
    return extract(g, all);
}

FeatureMatrix PageRank::extract(const CSRGraph& g,
                                 const std::vector<int64_t>& vertices) {
    int64_t V = g.num_vertices;
    double  d = damping_;
    double  init = 1.0 / V;

    std::vector<double> pr(V, init);
    std::vector<double> pr_new(V, 0.0);


    std::vector<int64_t> out_deg(V);
    for (int64_t v = 0; v < V; ++v)
        out_deg[v] = g.degree(v);

    for (int iter = 0; iter < max_iter_; ++iter) {

        double dangling_mass = 0.0;
        for (int64_t v = 0; v < V; ++v)
            if (out_deg[v] == 0)
                dangling_mass += pr[v];

        double base = (1.0 - d) / V + d * dangling_mass / V;



        std::fill(pr_new.begin(), pr_new.end(), base);

        #pragma omp parallel for schedule(guided) default(none) \
            shared(g, pr, pr_new, out_deg, d)
        for (int64_t v = 0; v < (int64_t)g.num_vertices; ++v) {
            if (out_deg[v] == 0) continue;
            double contrib = d * pr[v] / out_deg[v];
            for (int64_t e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
                int64_t w = g.col_idx[e];
                #pragma omp atomic
                pr_new[w] += contrib;
            }
        }


        double err = 0.0;
        #pragma omp parallel for reduction(+:err) schedule(static)
        for (int64_t v = 0; v < V; ++v)
            err += std::fabs(pr_new[v] - pr[v]);

        std::swap(pr, pr_new);

        if (err < tolerance_) {
            std::cout << "[PageRank] Converged at iteration " << iter + 1
                      << " (err=" << err << ")\n";
            break;
        }
    }


    int64_t n = static_cast<int64_t>(vertices.size());
    FeatureMatrix fm;
    fm.num_vertices = n;
    fm.num_features = 1;
    fm.names        = {"pagerank"};
    fm.data.resize(n);

    for (int64_t i = 0; i < n; ++i)
        fm.data[i] = static_cast<float>(pr[vertices[i]]);

    return fm;
}

}
