#include "core/graph_loader.h"
#include "core/csr_builder.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <iostream>

namespace gml {

CSRGraph GraphLoader::load(const std::string& path,
                            GraphFormat fmt,
                            bool directed,
                            bool weighted) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open graph file: " + path);

    CSRBuilder builder(directed);
    std::string line;
    int64_t edge_count = 0;

    while (std::getline(file, line)) {

        if (line.empty()) continue;
        if (fmt == GraphFormat::SNAP && line[0] == '#') continue;

        std::istringstream iss(line);
        int64_t src, dst;
        float   weight = 1.0f;

        if (!(iss >> src >> dst)) continue;
        if (weighted) iss >> weight;

        builder.add_edge(src, dst, weight);
        ++edge_count;
    }

    std::cout << "[GraphLoader] Loaded " << edge_count << " edges from " << path << "\n";
    return builder.build();
}

CSRGraph GraphLoader::generate_barabasi_albert(int64_t n, int m, uint64_t seed) {


    std::mt19937_64 rng(seed);
    CSRBuilder builder(false);


    for (int i = 0; i <= m; ++i)
        for (int j = i+1; j <= m; ++j)
            builder.add_edge(i, j);


    std::vector<int64_t> degree_list;
    for (int i = 0; i <= m; ++i)
        for (int k = 0; k < m; ++k)
            degree_list.push_back(i);

    for (int64_t v = m + 1; v < n; ++v) {
        std::vector<int64_t> targets;
        while (static_cast<int>(targets.size()) < m) {
            std::uniform_int_distribution<int64_t> dist(0, (int64_t)degree_list.size()-1);
            int64_t candidate = degree_list[dist(rng)];

            bool dup = false;
            for (auto t : targets) if (t == candidate) { dup = true; break; }
            if (!dup && candidate != v) targets.push_back(candidate);
        }
        for (auto t : targets) {
            builder.add_edge(v, t);
            degree_list.push_back(v);
            degree_list.push_back(t);
        }
    }

    std::cout << "[GraphLoader] Generated BA graph: n=" << n << " m=" << m << "\n";
    return builder.build();
}

CSRGraph GraphLoader::generate_erdos_renyi(int64_t n, double p, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    CSRBuilder builder(false);

    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i+1; j < n; ++j)
            if (dist(rng) < p)
                builder.add_edge(i, j);

    std::cout << "[GraphLoader] Generated ER graph: n=" << n << " p=" << p << "\n";
    return builder.build();
}

}
