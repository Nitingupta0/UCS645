#pragma once
#include "graph.h"
#include <string>

namespace gml {









enum class GraphFormat { EDGE_LIST, SNAP, METIS };

class GraphLoader {
public:

    static CSRGraph load(const std::string& path,
                         GraphFormat fmt  = GraphFormat::SNAP,
                         bool directed    = false,
                         bool weighted    = false);



    static CSRGraph generate_barabasi_albert(int64_t n, int m = 3,
                                              uint64_t seed = 42);


    static CSRGraph generate_erdos_renyi(int64_t n, double p = 0.01,
                                          uint64_t seed = 42);
};

}
