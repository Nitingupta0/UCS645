#include "core/csr_builder.h"
#include <algorithm>
#include <stdexcept>

namespace gml {

void CSRBuilder::add_edge(int64_t src, int64_t dst, float weight) {
    if (src < 0 || dst < 0)
        throw std::invalid_argument("Negative vertex index");
    edge_list_.emplace_back(src, dst, weight);
    max_vertex_ = std::max(max_vertex_, std::max(src, dst));
    if (!directed_) {
        edge_list_.emplace_back(dst, src, weight);
    }
}

void CSRBuilder::add_edges(
    const std::vector<std::tuple<int64_t,int64_t,float>>& edges) {
    edge_list_.reserve(edge_list_.size() + edges.size() * (directed_ ? 1 : 2));
    for (auto& [s, d, w] : edges) {
        add_edge(s, d, w);
    }
}

CSRGraph CSRBuilder::build() {
    if (edge_list_.empty()) return CSRGraph{};


    std::sort(edge_list_.begin(), edge_list_.end(),
              [](const auto& a, const auto& b){
                  return std::get<0>(a) < std::get<0>(b) ||
                         (std::get<0>(a) == std::get<0>(b) &&
                          std::get<1>(a) < std::get<1>(b));
              });


    edge_list_.erase(
        std::unique(edge_list_.begin(), edge_list_.end(),
                    [](const auto& a, const auto& b){
                        return std::get<0>(a) == std::get<0>(b) &&
                               std::get<1>(a) == std::get<1>(b);
                    }),
        edge_list_.end());

    CSRGraph g;
    g.directed    = directed_;
    g.num_vertices = max_vertex_ + 1;
    g.num_edges   = static_cast<int64_t>(edge_list_.size());

    g.row_ptr.assign(g.num_vertices + 1, 0);
    g.col_idx.resize(g.num_edges);
    g.weights.resize(g.num_edges);


    for (auto& [s, d, w] : edge_list_)
        g.row_ptr[s + 1]++;


    for (int64_t v = 0; v < g.num_vertices; ++v)
        g.row_ptr[v + 1] += g.row_ptr[v];


    std::vector<int64_t> cursor(g.num_vertices, 0);
    for (auto& [s, d, w] : edge_list_) {
        int64_t pos = g.row_ptr[s] + cursor[s]++;
        g.col_idx[pos] = d;
        g.weights[pos] = w;
    }

    return g;
}

void CSRBuilder::reset() {
    edge_list_.clear();
    max_vertex_ = -1;
}

}
