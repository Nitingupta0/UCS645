#include "test_harness.h"
#include "core/csr_builder.h"
#include "core/graph_loader.h"
#include <iostream>
#include <cassert>
#include <fstream>




using namespace gml;

void register_csr_tests() {
    std::cout << "-- CSR Builder tests --\n";


    {
        CSRBuilder b(false);
        b.add_edge(0,1); b.add_edge(1,2); b.add_edge(0,2);
        auto g = b.build();
        ASSERT_TRUE(g.num_vertices == 3);
        ASSERT_TRUE(g.num_edges   == 6);
        ASSERT_TRUE(g.degree(0)   == 2);
        ASSERT_TRUE(g.degree(1)   == 2);
        ASSERT_TRUE(g.degree(2)   == 2);
        std::cout << "  undirected triangle: ok\n";
    }


    {
        CSRBuilder b(true);
        b.add_edge(0,1); b.add_edge(1,2);
        auto g = b.build();
        ASSERT_TRUE(g.num_vertices == 3);
        ASSERT_TRUE(g.num_edges   == 2);
        ASSERT_TRUE(g.degree(0)   == 1);
        ASSERT_TRUE(g.degree(2)   == 0);
        std::cout << "  directed path: ok\n";
    }


    {
        CSRBuilder b(true);
        b.add_edge(0,1); b.add_edge(0,1); b.add_edge(0,1);
        auto g = b.build();
        ASSERT_TRUE(g.num_edges == 1);
        std::cout << "  deduplication: ok\n";
    }


    {
        auto g = GraphLoader::generate_barabasi_albert(100, 3, 42);
        ASSERT_TRUE(g.num_vertices == 100);
        ASSERT_TRUE(g.num_edges > 0);
        std::cout << "  BA graph generation: ok\n";
    }


    {
        auto g = GraphLoader::generate_erdos_renyi(50, 0.3, 42);
        ASSERT_TRUE(g.num_vertices == 50);
        std::cout << "  ER graph generation: ok\n";
    }


    {
        CSRBuilder b(false);
        b.add_edge(0,1); b.add_edge(0,2); b.add_edge(0,3);
        auto g = b.build();
        ASSERT_TRUE(g.degree(0) == 3);
        for (auto it = g.neighbors_begin(0); it != g.neighbors_end(0); ++it)
            ASSERT_TRUE(*it >= 0 && *it < g.num_vertices);
        std::cout << "  neighbour iterator bounds: ok\n";
    }


    {
        const std::string path = "/tmp/gml_graph_loader_test.txt";
        {
            std::ofstream out(path);
            out << "# tiny undirected triangle\n";
            out << "0 1\n";
            out << "1 2\n";
            out << "2 0\n";
        }

        auto g = GraphLoader::load(path, GraphFormat::SNAP, false);
        ASSERT_TRUE(g.num_vertices == 3);
        ASSERT_TRUE(g.num_edges == 6);
        ASSERT_TRUE(g.degree(0) == 2);
        ASSERT_TRUE(g.degree(1) == 2);
        ASSERT_TRUE(g.degree(2) == 2);
        std::cout << "  file-based graph loading (SNAP): ok\n";
    }
}
