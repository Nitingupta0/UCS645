#include <algorithm>
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <iomanip>
#include <cmath>

#include "core/graph.h"
#include "core/graph_loader.h"
#include "core/workload_splitter.h"
#include "features/degree_centrality.h"
#include "features/betweenness_centrality.h"
#include "features/pagerank.h"
#include "features/clustering_coefficient.h"
#include "features/feature_aggregator.h"
#include "ml/random_forest.h"
#include "ml/evaluator.h"

#ifdef ENABLE_CUDA
#include "gpu/feature_normalizer.h"
#include "gpu/i_gpu_processor.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace gml;
using Clock = std::chrono::high_resolution_clock;





static Dataset make_dataset(const CSRGraph& g,
                              const FeatureMatrix& fm,
                              int num_classes = 5) {
    int64_t V = g.num_vertices;


    std::vector<double> log_deg(V);
    double min_ld =  1e30, max_ld = -1e30;
    for (int64_t v = 0; v < V; ++v) {
        log_deg[v] = std::log1p(static_cast<double>(g.degree(v)));
        if (log_deg[v] < min_ld) min_ld = log_deg[v];
        if (log_deg[v] > max_ld) max_ld = log_deg[v];
    }
    double range = max_ld - min_ld + 1e-9;


    std::vector<int64_t> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b){ return log_deg[a] < log_deg[b]; });

    std::vector<int> labels(V);
    for (int64_t i = 0; i < V; ++i)
        labels[order[i]] = static_cast<int>(i * num_classes / V);

    Dataset ds;
    ds.features    = fm;
    ds.labels      = labels;
    ds.num_classes = num_classes;
    return ds;
}




static void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║   Scalable Graph Feature Extraction & ML Pipeline           ║
║   Hybrid CPU (OpenMP) + GPU (CUDA)  |  Random Forest        ║
╚══════════════════════════════════════════════════════════════╝
)" << "\n";
}




static void print_help() {
    std::cout << "Usage: graph_ml [OPTIONS]\n\n"
              << "Options:\n"
              << "  --vertices N       Number of vertices for synthetic graph (default: 10000)\n"
              << "  --trees N          Number of Random Forest trees (default: 50)\n"
              << "  --bc-samples N     Betweenness centrality sample count (default: 200)\n"
              << "  --gpu              Enable CUDA GPU processing\n"
              << "  --sequential       Force single-threaded execution (OpenMP threads = 1)\n"
              << "  --threads N        Set OpenMP thread count explicitly\n"
              << "  --input FILE       Load graph from SNAP edge-list file\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char* argv[]) {
    print_banner();


    int64_t     n_vertices   = 10000;
    int         n_trees      = 50;
    int         n_samples_bc = 200;
    bool        use_gpu      = false;
    int         n_threads    = -1;
    std::string input_file;


    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help")                                 { print_help(); return 0; }
        if (arg == "--vertices"   && i+1 < argc)   n_vertices   = std::stoll(argv[++i]);
        if (arg == "--trees"      && i+1 < argc)   n_trees      = std::stoi (argv[++i]);
        if (arg == "--bc-samples" && i+1 < argc)   n_samples_bc = std::stoi (argv[++i]);
        if (arg == "--gpu")                        use_gpu      = true;
        if (arg == "--sequential")                 n_threads    = 1;
        if (arg == "--threads"    && i+1 < argc)   n_threads    = std::stoi (argv[++i]);
        if (arg == "--input"      && i+1 < argc)   input_file   = argv[++i];
    }

#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    std::cout << "[Config] OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "[Config] OpenMP: disabled\n";
#endif

#ifdef ENABLE_CUDA
    if (use_gpu && IGpuProcessor::cuda_available()) {
        std::cout << "[Config] CUDA: enabled  ("
                  << IGpuProcessor::available_vram() / (1<<20) << " MB free VRAM)\n";
    } else {
        use_gpu = false;
        std::cout << "[Config] CUDA: not available, running CPU-only\n";
    }
#else
    use_gpu = false;
    std::cout << "[Config] CUDA: not compiled in\n";
#endif

    std::cout << "[Config] Graph size: " << n_vertices << " vertices\n";
    std::cout << "[Config] Random Forest: " << n_trees << " trees\n\n";


    auto t0 = Clock::now();
    std::cout << "━━ Phase 1: Graph Generation ━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    CSRGraph g;
    if (!input_file.empty()) {
        std::cout << "[Loader] Loading graph from: " << input_file << "\n";
        g = GraphLoader::load(input_file);
    } else {
        g = GraphLoader::generate_barabasi_albert(n_vertices, 3);
    }

    double ms_ingest = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    Evaluator::print_timing("Graph generation", ms_ingest);
    std::cout << "   Vertices: " << g.num_vertices
              << "  Edges: " << g.num_edges << "\n\n";


    std::cout << "━━ Phase 2: Workload Partitioning ━━━━━━━━━━━━━━━━━━━━\n";
    WorkloadSplitter splitter;
    WorkloadPartition partition = splitter.split_adaptive(g);
    std::cout << "\n";


    auto t1 = Clock::now();
    std::cout << "━━ Phase 3: Feature Extraction (CPU / OpenMP) ━━━━━━━━\n";

    FeatureAggregator aggregator;
    aggregator.add_extractor(std::make_unique<DegreeCentrality>());
    aggregator.add_extractor(
        std::make_unique<BetweennessCentrality>(n_samples_bc));
    aggregator.add_extractor(std::make_unique<PageRank>());
    aggregator.add_extractor(std::make_unique<ClusteringCoefficient>());

    FeatureMatrix fm = aggregator.aggregate(g);
    double ms_feat = std::chrono::duration<double,std::milli>(Clock::now()-t1).count();
    Evaluator::print_timing("Feature extraction", ms_feat);
    std::cout << "   Feature matrix: " << fm.num_vertices
              << " × " << fm.num_features << "\n";
    std::cout << "   Features: [";
    for (size_t i = 0; i < fm.names.size(); ++i)
        std::cout << fm.names[i] << (i+1<fm.names.size() ? ", " : "");
    std::cout << "]\n\n";


    auto t2 = Clock::now();
    std::cout << "━━ Phase 4: Feature Preprocessing";
#ifdef ENABLE_CUDA
    if (use_gpu) {
        std::cout << " (CUDA) ━━━━━━━━━━━━━━━━\n";
        FeatureNormalizerCUDA normalizer(NormMode::Z_SCORE);
        normalizer.process(fm);
    } else {
#endif
        std::cout << " (CPU fallback) ━━━━━━━━━\n";

        int64_t N = fm.num_vertices, F = fm.num_features;
        for (int64_t f = 0; f < F; ++f) {
            double mean = 0;
            #pragma omp parallel for reduction(+:mean) schedule(static)
            for (int64_t v = 0; v < N; ++v) mean += fm.at(v, f);
            mean /= N;

            double var = 0;
            #pragma omp parallel for reduction(+:var) schedule(static)
            for (int64_t v = 0; v < N; ++v)
                var += (fm.at(v,f)-mean)*(fm.at(v,f)-mean);
            var /= N;

            float std_inv = 1.0f / (std::sqrt((float)var) + 1e-8f);
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; ++v)
                fm.at(v, f) = (fm.at(v, f) - (float)mean) * std_inv;
        }
        std::cout << "[Normalizer] CPU z-score applied to "
                  << N << "×" << F << " matrix\n";
#ifdef ENABLE_CUDA
    }
#endif
    double ms_preproc = std::chrono::duration<double,std::milli>(Clock::now()-t2).count();
    Evaluator::print_timing("Feature preprocessing", ms_preproc);
    std::cout << "\n";


    std::cout << "━━ Phase 5: Dataset Preparation ━━━━━━━━━━━━━━━━━━━━━\n";
    int num_classes = 5;
    Dataset ds = make_dataset(g, fm, num_classes);


    auto [train_ds, test_ds] = ds.train_test_split(0.2, 42);

    std::cout << "   Train: " << train_ds.features.num_vertices
              << "  Test: " << test_ds.features.num_vertices
              << "  Classes: " << num_classes << "\n\n";


    auto t3 = Clock::now();
    std::cout << "━━ Phase 6: Random Forest Training ━━━━━━━━━━━━━━━━━━━\n";
    ParallelRandomForest rf(n_trees, 10, 5,
                             0.8, -1, 42);
    rf.fit(train_ds);
    double ms_train = std::chrono::duration<double,std::milli>(Clock::now()-t3).count();
    Evaluator::print_timing("Training", ms_train);


    auto importances = rf.feature_importances();
    std::cout << "\n   Feature Importances (MDI):\n";
    for (size_t i = 0; i < importances.size() && i < fm.names.size(); ++i) {
        std::cout << "     " << std::setw(24) << std::left << fm.names[i]
                  << std::fixed << std::setprecision(4) << importances[i] << "\n";
    }
    std::cout << "\n";


    auto t4 = Clock::now();
    std::cout << "━━ Phase 7: Inference & Evaluation ━━━━━━━━━━━━━━━━━━━\n";
    auto predictions = rf.predict(test_ds.features);
    double ms_infer = std::chrono::duration<double,std::milli>(Clock::now()-t4).count();
    Evaluator::print_timing("Inference", ms_infer);

    Evaluator eval(num_classes);
    ClassificationReport report = eval.evaluate(test_ds.labels, predictions);
    std::cout << report.to_string();


    double total = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    std::cout << "━━ Pipeline Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "   Graph ingest:    " << ms_ingest  << " ms\n";
    std::cout << "   Feature extract: " << ms_feat    << " ms\n";
    std::cout << "   Preprocessing:   " << ms_preproc << " ms\n";
    std::cout << "   RF training:     " << ms_train   << " ms\n";
    std::cout << "   Inference:       " << ms_infer   << " ms\n";
    std::cout << "   ─────────────────────────────\n";
    std::cout << "   Total:           " << total      << " ms\n";
    std::cout << "   Accuracy:        " << report.accuracy * 100 << " %\n\n";


    int threads_used = 1;
#ifdef _OPENMP
    threads_used = omp_get_max_threads();
#endif
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[RESULT] "
              << "threads=" << threads_used
              << ",vertices=" << g.num_vertices
              << ",edges=" << g.num_edges
              << ",ingest_ms=" << ms_ingest
              << ",features_ms=" << ms_feat
              << ",preproc_ms=" << ms_preproc
              << ",train_ms=" << ms_train
              << ",infer_ms=" << ms_infer
              << ",total_ms=" << total
              << ",accuracy=" << report.accuracy
              << ",f1=" << report.macro_f1
              << "\n";

    return 0;
}
