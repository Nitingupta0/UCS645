#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace gml {

struct CSRGraph {
    int64_t num_vertices = 0;
    int64_t num_edges    = 0;

    std::vector<int64_t> row_ptr;
    std::vector<int64_t> col_idx;
    std::vector<float>   weights;

    bool directed = false;

    int64_t degree(int64_t v) const {
        return row_ptr[v + 1] - row_ptr[v];
    }

    const int64_t* neighbors_begin(int64_t v) const {
        return col_idx.data() + row_ptr[v];
    }

    const int64_t* neighbors_end(int64_t v) const {
        return col_idx.data() + row_ptr[v + 1];
    }
};

class IGraph {
public:
    virtual ~IGraph() = default;
    virtual int64_t num_vertices() const = 0;
    virtual int64_t num_edges()    const = 0;
    virtual const CSRGraph& csr()  const = 0;
};

}
