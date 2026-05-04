#include "test_harness.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <iostream>
#include "ml/decision_tree.h"
#include "ml/random_forest.h"
#include "ml/evaluator.h"
#include "core/graph_loader.h"
#include "features/degree_centrality.h"
#include "features/feature_aggregator.h"




using namespace gml;


static Dataset make_linear(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(0.f, 1.f);
    FeatureMatrix fm;
    fm.num_vertices = n; fm.num_features = 2;
    fm.names = {"x","y"};
    fm.data.resize(n * 2);
    std::vector<int> labels(n);
    for (int i = 0; i < n; ++i) {
        float x = d(rng), y = d(rng);
        fm.data[i*2] = x; fm.data[i*2+1] = y;
        labels[i] = (x + y > 1.f) ? 1 : 0;
    }
    Dataset ds; ds.features = fm; ds.labels = labels; ds.num_classes = 2;
    return ds;
}

void register_ml_tests() {
    std::cout << "\n-- ML tests --\n";


    {
        auto ds = make_linear(100, 42);
        DecisionTree dt(20, 1);
        dt.fit(ds);
        auto preds = dt.predict(ds.features);
        int correct = 0;
        for (int i = 0; i < (int)ds.labels.size(); ++i)
            if (preds[i] == ds.labels[i]) ++correct;
        ASSERT_TRUE(correct >= 90);
        std::cout << "  decision tree train accuracy >= 90% (" << correct << "/100): ok\n";
    }


    {
        auto train = make_linear(400, 42);
        auto test  = make_linear(100, 999);
        ParallelRandomForest rf(20, 8, 3, 0.8, -1, 7);
        rf.fit(train);
        auto preds = rf.predict(test.features);
        int correct = 0;
        for (int i = 0; i < (int)test.labels.size(); ++i)
            if (preds[i] == test.labels[i]) ++correct;
        ASSERT_TRUE(correct >= 75);
        std::cout << "  random forest test accuracy >= 75% (" << correct << "/100): ok\n";
    }


    {
        std::vector<int> y_true = {0,0,1,1,2,2};
        std::vector<int> y_pred = {0,1,1,1,2,0};
        Evaluator ev(3);
        auto r = ev.evaluate(y_true, y_pred);
        ASSERT_TRUE(r.confusion_matrix[0][0] == 1);
        ASSERT_TRUE(r.confusion_matrix[1][1] == 2);
        ASSERT_TRUE(r.confusion_matrix[2][2] == 1);
        ASSERT_NEAR(r.accuracy, 4.0/6.0, 1e-5);
        std::cout << "  evaluator confusion matrix: ok\n";
    }


    {
        auto ds = make_linear(100, 42);
        ParallelRandomForest rf(10, 5, 3, 0.8, -1, 42);
        rf.fit(ds);
        auto proba = rf.predict_proba(ds.features, 0);
        bool all_valid = true;
        for (float p : proba) if (p < 0.f || p > 1.f) { all_valid = false; break; }
        ASSERT_TRUE(all_valid);
        std::cout << "  predict_proba in [0,1]: ok\n";
    }


    {
        auto ds = make_linear(50, 123);
        auto [train_a, test_a] = ds.train_test_split(0.2, 7);
        auto [train_b, test_b] = ds.train_test_split(0.2, 7);

        ASSERT_TRUE(train_a.features.num_vertices == 40);
        ASSERT_TRUE(test_a.features.num_vertices == 10);
        ASSERT_TRUE(train_a.features.num_features == ds.features.num_features);
        ASSERT_TRUE(test_a.features.num_features == ds.features.num_features);
        ASSERT_TRUE(train_a.labels.size() == (size_t)train_a.features.num_vertices);
        ASSERT_TRUE(test_a.labels.size() == (size_t)test_a.features.num_vertices);

        ASSERT_TRUE(train_a.labels == train_b.labels);
        ASSERT_TRUE(test_a.labels == test_b.labels);
        ASSERT_TRUE(train_a.features.data == train_b.features.data);
        ASSERT_TRUE(test_a.features.data == test_b.features.data);

        ASSERT_TRUE(train_a.features.num_vertices + test_a.features.num_vertices
                    == ds.features.num_vertices);
        std::cout << "  dataset train_test_split deterministic and sized: ok\n";
    }


    {
        auto ds = make_linear(300, 2024);
        ParallelRandomForest rf(20, 8, 3, 0.8, -1, 99);
        rf.fit(ds);

        auto imp = rf.feature_importances();
        ASSERT_TRUE(imp.size() == (size_t)ds.features.num_features);

        float sum = 0.0f;
        bool valid = true;
        for (float v : imp) {
            sum += v;
            if (!(v >= 0.0f && v <= 1.0f) || !std::isfinite(v)) {
                valid = false;
                break;
            }
        }
        ASSERT_TRUE(valid);
        ASSERT_NEAR(sum, 1.0f, 1e-3f);
        std::cout << "  feature importances normalized and valid: ok\n";
    }


    {
        std::vector<int> y_true = {0,0,0,1,1,1};
        std::vector<int> y_pred = {0,0,1,1,1,1};
        Evaluator ev(2);
        auto r = ev.evaluate(y_true, y_pred);
        double expected_acc = 5.0 / 6.0;
        ASSERT_NEAR(r.accuracy, expected_acc, 1e-5);
        std::cout << "  evaluator accuracy 5/6: ok\n";
    }


    {
        auto g = GraphLoader::generate_barabasi_albert(500, 3, 42);
        FeatureAggregator agg;
        agg.add_extractor(std::make_unique<DegreeCentrality>());
        auto fm = agg.aggregate(g);

        int64_t V = g.num_vertices;
        std::vector<int64_t> order(V);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int64_t a, int64_t b){
                      return std::log1p(g.degree(a)) < std::log1p(g.degree(b));
                  });
        std::vector<int> labels(V);
        for (int64_t i = 0; i < V; ++i) labels[order[i]] = (int)(i * 3 / V);

        Dataset ds; ds.features = fm; ds.labels = labels; ds.num_classes = 3;

        int64_t n_train = V * 4 / 5;
        auto make_sub = [&](int64_t s, int64_t e) {
            Dataset d; d.num_classes = 3;
            d.features.num_vertices = e - s;
            d.features.num_features = fm.num_features;
            d.features.names = fm.names;
            d.features.data.assign(fm.data.begin() + s * fm.num_features,
                                    fm.data.begin() + e * fm.num_features);
            d.labels.assign(labels.begin() + s, labels.begin() + e);
            return d;
        };

        ParallelRandomForest rf(30, 8, 3, 0.8, -1, 42);
        rf.fit(make_sub(0, n_train));
        auto preds = rf.predict(make_sub(n_train, V).features);

        Evaluator ev(3);
        auto r = ev.evaluate(std::vector<int>(labels.begin()+n_train, labels.end()), preds);
        ASSERT_TRUE(r.accuracy > 0.5);
        std::cout << "  end-to-end graph->RF accuracy=" << r.accuracy*100 << "%: ok\n";
    }
}
