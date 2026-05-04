#include "test_harness.h"
#include "core/csr_builder.h"
#include "core/graph_loader.h"
#include "features/degree_centrality.h"
#include "features/pagerank.h"
#include "features/clustering_coefficient.h"
#include "features/betweenness_centrality.h"
#include "features/feature_aggregator.h"
#include <iostream>
#include <cmath>
#include <numeric>




using namespace gml;

void register_feature_tests() {
    std::cout << "\n-- Feature extraction tests --\n";


    CSRBuilder sb(false);
    for (int i = 1; i <= 4; ++i) sb.add_edge(0, i);
    auto star = sb.build();


    {
        DegreeCentrality dc;
        auto fm = dc.extract(star);
        ASSERT_TRUE(fm.num_features == 1);
        ASSERT_NEAR(fm.at(0, 0), 1.0f, 1e-5f);
        ASSERT_NEAR(fm.at(1, 0), 0.25f, 1e-5f);
        std::cout << "  degree centrality (star): ok\n";
    }


    {
        ClusteringCoefficient cc;
        auto fm = cc.extract(star);
        ASSERT_NEAR(fm.at(0, 0), 0.0f, 1e-5f);
        std::cout << "  clustering coeff (star center=0): ok\n";
    }


    {
        CSRBuilder tb(false);
        tb.add_edge(0,1); tb.add_edge(1,2); tb.add_edge(0,2);
        auto tri = tb.build();
        ClusteringCoefficient cc;
        auto fm = cc.extract(tri);
        for (int v = 0; v < 3; ++v)
            ASSERT_NEAR(fm.at(v, 0), 1.0f, 1e-4f);
        std::cout << "  clustering coeff (triangle=1): ok\n";
    }


    {
        auto g = GraphLoader::generate_erdos_renyi(20, 0.3, 42);
        PageRank pr;
        auto fm = pr.extract(g);
        double sum = 0;
        for (int64_t v = 0; v < fm.num_vertices; ++v) sum += fm.at(v, 0);
        ASSERT_NEAR(sum, 1.0, 1e-3);
        std::cout << "  PageRank sum=1: ok\n";
    }


    {
        FeatureAggregator agg;
        agg.add_extractor(std::make_unique<DegreeCentrality>());
        agg.add_extractor(std::make_unique<PageRank>());
        auto fm = agg.aggregate(star);
        ASSERT_TRUE(fm.num_features == 2);
        ASSERT_TRUE(fm.num_vertices == star.num_vertices);
        std::cout << "  feature aggregator (2 extractors): ok\n";
    }


    {
        BetweennessCentrality bc(0);
        auto fm = bc.extract(star);
        float center_bc = fm.at(0, 0);
        bool center_is_max = true;
        for (int v = 1; v < (int)star.num_vertices; ++v)
            if (fm.at(v, 0) > center_bc) { center_is_max = false; break; }
        ASSERT_TRUE(center_is_max);
        std::cout << "  betweenness centrality (star center=max): ok\n";
    }


    {
        DegreeCentrality dc;
        auto fm = dc.extract(star);
        ASSERT_TRUE(fm.names.size() == 1);
        ASSERT_TRUE(fm.names[0] == "degree_centrality");
        std::cout << "  feature names populated: ok\n";
    }
}
