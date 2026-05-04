#include "features/betweenness_centrality.h"
#include <queue>
#include <stack>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gml {





void BetweennessCentrality::brandes_bfs(const CSRGraph& g,
                                         int64_t source,
                                         std::vector<double>& delta,
                                         std::vector<double>& bc_local) const {
    int64_t V = g.num_vertices;

    std::vector<std::vector<int64_t>> pred(V);
    std::vector<double> sigma(V, 0.0);
    std::vector<int64_t> dist(V, -1);

    sigma[source] = 1.0;
    dist[source]  = 0;

    std::queue<int64_t> Q;
    std::stack<int64_t> S;
    Q.push(source);


    while (!Q.empty()) {
        int64_t v = Q.front(); Q.pop();
        S.push(v);
        for (int64_t e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
            int64_t w = g.col_idx[e];
            if (dist[w] < 0) {
                dist[w] = dist[v] + 1;
                Q.push(w);
            }
            if (dist[w] == dist[v] + 1) {
                sigma[w] += sigma[v];
                pred[w].push_back(v);
            }
        }
    }


    std::fill(delta.begin(), delta.end(), 0.0);
    while (!S.empty()) {
        int64_t w = S.top(); S.pop();
        for (int64_t v : pred[w]) {
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
        if (w != source)
            bc_local[w] += delta[w];
    }
}

FeatureMatrix BetweennessCentrality::extract(const CSRGraph& g) {
    std::vector<int64_t> all(g.num_vertices);
    std::iota(all.begin(), all.end(), 0);
    return extract(g, all);
}

FeatureMatrix BetweennessCentrality::extract(const CSRGraph& g,
                                              const std::vector<int64_t>& vertices) {
    int64_t V = g.num_vertices;
    int64_t n = static_cast<int64_t>(vertices.size());


    std::vector<int64_t> sources;
    if (num_samples_ <= 0 || num_samples_ >= V) {
        sources.resize(V);
        std::iota(sources.begin(), sources.end(), 0);
    } else {
        std::mt19937_64 rng(seed_);
        sources.resize(V);
        std::iota(sources.begin(), sources.end(), 0);
        std::shuffle(sources.begin(), sources.end(), rng);
        sources.resize(num_samples_);
    }


    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    std::vector<std::vector<double>> bc_per_thread(nthreads,
                                                     std::vector<double>(V, 0.0));

    #pragma omp parallel default(none) shared(g, sources, bc_per_thread, V)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        std::vector<double> delta(V, 0.0);

        #pragma omp for schedule(dynamic, 1)
        for (int64_t si = 0; si < (int64_t)sources.size(); ++si) {
            brandes_bfs(g, sources[si], delta, bc_per_thread[tid]);
        }
    }


    std::vector<double> bc(V, 0.0);
    for (auto& tbc : bc_per_thread)
        for (int64_t v = 0; v < V; ++v)
            bc[v] += tbc[v];


    double norm = 1.0;
    if (normalise_ && V > 2) {
        double factor = g.directed
            ? (double)(V - 1) * (V - 2)
            : (double)(V - 1) * (V - 2) / 2.0;

        double sample_ratio = (double)sources.size() / V;
        norm = factor * sample_ratio;
    }


    FeatureMatrix fm;
    fm.num_vertices = n;
    fm.num_features = 1;
    fm.names        = {"betweenness_centrality"};
    fm.data.resize(n);

    for (int64_t i = 0; i < n; ++i)
        fm.data[i] = static_cast<float>(bc[vertices[i]] / norm);

    return fm;
}

}
