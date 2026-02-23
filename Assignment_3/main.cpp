/*
 * main.cpp — Single entry point for all three correlation experiments.
 *
 * Compile with -DQUES1 → ques1 (Sequential Baseline)
 *              -DQUES2 → ques2 (Thread Scaling + Scheduling)
 *              -DQUES3 → ques3 (Optimized vs Basic + Stats)
 *
 * Each question links against its own correlate implementation:
 *   ques1 ← correlate1.cpp  (pure sequential)
 *   ques2 ← correlate2.cpp  (basic OpenMP parallel)
 *   ques3 ← correlate3.cpp  (SIMD + ILP optimized)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sys/resource.h>
#include <omp.h>

/* Provided by the linked correlatex.cpp */
extern void correlate(int ny, int nx, const float *data, float *result);

/* ------------------------------------------------------------------ */
/*  Shared utilities                                                    */
/* ------------------------------------------------------------------ */

static void generate_data(float *data, int ny, int nx)
{
    unsigned int seed = 42;
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
            data[y * nx + x] = (float)rand_r(&seed) / RAND_MAX * 2.0f - 1.0f;
}

/* Pure sequential kernel — used as baseline in ques2 & ques3 */
static void correlate_seq(int ny, int nx, const float *data, float *result)
{
    double *norm = (double *)malloc((size_t)ny * nx * sizeof(double));
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) sum += data[y * nx + x];
        double mean = sum / nx;
        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = data[y * nx + x] - mean;
            norm[y * nx + x] = val;
            sq_sum += val * val;
        }
        double inv_len = 1.0 / sqrt(sq_sum);
        for (int x = 0; x < nx; x++) norm[y * nx + x] *= inv_len;
    }
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[i * nx + x] * norm[j * nx + x];
            result[i + j * ny] = (float)dot;
        }
    }
    free(norm);
}

/* Basic parallel kernel (used in ques3 for comparison) */
static void correlate_basic(int ny, int nx, const float *data, float *result)
{
    double *norm = (double *)malloc((size_t)ny * nx * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) sum += data[y * nx + x];
        double mean = sum / nx;
        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = data[y * nx + x] - mean;
            norm[y * nx + x] = val;
            sq_sum += val * val;
        }
        double inv_len = 1.0 / sqrt(sq_sum);
        for (int x = 0; x < nx; x++) norm[y * nx + x] *= inv_len;
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[i * nx + x] * norm[j * nx + x];
            result[i + j * ny] = (float)dot;
        }
    }
    free(norm);
}

/* Runtime-scheduled kernel — used in ques2 scheduling comparison */
static void correlate_runtime(int ny, int nx, const float *data, float *result)
{
    double *norm = (double *)malloc((size_t)ny * nx * sizeof(double));

    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) sum += data[y * nx + x];
        double mean = sum / nx;
        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = data[y * nx + x] - mean;
            norm[y * nx + x] = val;
            sq_sum += val * val;
        }
        double inv_len = 1.0 / sqrt(sq_sum);
        for (int x = 0; x < nx; x++) norm[y * nx + x] *= inv_len;
    }

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[i * nx + x] * norm[j * nx + x];
            result[i + j * ny] = (float)dot;
        }
    }
    free(norm);
}

/* ================================================================== */
/*  QUES1 — Sequential Correlation Matrix Baseline                     */
/* ================================================================== */
#ifdef QUES1

int main()
{
    printf("Physical Cores: 10\n");
    printf("Logical Cores : 12 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    printf("=== Sequential Correlation Matrix Baseline (Double Precision) ===\n\n");
    printf("%-8s | %-8s | %-14s | %-14s | %-12s\n",
           "ny", "nx", "Time (s)", "GFLOP/s", "Max Diag Err");
    printf("---------|----------|----------------|----------------|------------\n");

    int test_ny[] = {300, 600, 1000};
    int test_nx[] = {3000, 5000, 8000};

    for (int s = 0; s < 3; s++) {
        int ny = test_ny[s], nx = test_nx[s];

        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        double t1 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double secs = omp_get_wtime() - t1;

        double pairs  = (double)ny * (ny + 1) / 2.0;
        double gflops = pairs * 2.0 * nx / 1e9 / secs;

        double max_err = 0.0;
        for (int i = 0; i < ny; i++) {
            double e = fabs((double)result[i + i * ny] - 1.0);
            if (e > max_err) max_err = e;
        }

        printf("%-8d | %-8d | %-14.6f | %-14.2f | %.2e\n",
               ny, nx, secs, gflops, max_err);

        free(data); free(result);
    }

    /* Detailed analysis for ny=600, nx=5000 */
    printf("\n=== Detailed Analysis for ny=600, nx=5000 ===\n\n");
    {
        int ny = 600, nx = 5000;
        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        double t1 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double seq_time = omp_get_wtime() - t1;

        printf("Matrix dimensions: %d rows x %d columns\n", ny, nx);
        printf("Unique row-pairs computed: %lld\n", (long long)ny * (ny + 1) / 2);
        printf("Sequential time: %f s\n", seq_time);
        printf("Data memory: %.2f MB\n",    (double)ny * nx * sizeof(float)  / (1<<20));
        printf("Result memory: %.2f MB\n",  (double)ny * ny * sizeof(float)  / (1<<20));
        printf("Working memory (double): %.2f MB\n",
                                            (double)ny * nx * sizeof(double) / (1<<20));

        printf("\nSample diagonal values (should be 1.0):\n");
        printf("%-10s | %-16s\n", "Row", "corr(i,i)");
        printf("-----------|------------------\n");
        for (int i = 0; i < 5; i++)
            printf("%-10d | %.12f\n", i, (double)result[i + i * ny]);

        printf("\nSample off-diagonal values:\n");
        printf("%-8s | %-8s | %-16s\n", "Row i", "Row j", "corr(i,j)");
        printf("---------|---------|------------------\n");
        int si[] = {100, 200, 300, 400, 500};
        int sj[] = { 50, 100, 150, 200, 250};
        for (int k = 0; k < 5; k++)
            printf("%-8d | %-8d | %.12f\n",
                   si[k], sj[k], (double)result[si[k] + sj[k] * ny]);

        free(data); free(result);
    }

    return 0;
}

/* ================================================================== */
/*  QUES2 — Thread Scaling + Scheduling Strategy + Performance Stats   */
/* ================================================================== */
#elif defined(QUES2)

int main()
{
    printf("Physical Cores: 10\n");
    printf("Logical Cores : 12 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    /* ---- Part 1: Thread Scaling ---- */
    printf("=== Part 1: Thread Scaling (Static Schedule) ===\n\n");
    printf("%-8s | %-8s | %-8s | %-14s | %-14s | %-10s\n",
           "ny", "nx", "Threads", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|----------|----------|----------------|----------------|----------\n");

    int test_ny[] = {300, 600, 1000};
    int test_nx[] = {3000, 5000, 8000};

    for (int s = 0; s < 3; s++) {
        int ny = test_ny[s], nx = test_nx[s];

        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(1);
        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t1 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double seq_time = omp_get_wtime() - t1;

        for (int t = 1; t <= 10; t++) {
            omp_set_num_threads(t);
            memset(result, 0, (size_t)ny * ny * sizeof(float));
            double t3 = omp_get_wtime();
            correlate(ny, nx, data, result);
            double par_time = omp_get_wtime() - t3;
            printf("%-8d | %-8d | %-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   ny, nx, t, seq_time, par_time, seq_time / par_time);
        }
        printf("---------|----------|----------|----------------|----------------|----------\n");

        free(data); free(result);
    }

    /* ---- Part 2: Scheduling Strategy Comparison ---- */
    printf("\n=== Part 2: Scheduling Strategy Comparison (4 threads, ny=800, nx=5000) ===\n\n");
    printf("%-12s | %-14s | %-14s | %-10s\n",
           "Schedule", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("-------------|----------------|----------------|----------\n");
    {
        int ny = 800, nx = 5000;
        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(1);
        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t1 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double seq_time = omp_get_wtime() - t1;

        omp_set_num_threads(4);

        struct { const char *name; omp_sched_t kind; int chunk; } scheds[] = {
            {"static",     omp_sched_static,  0},
            {"dynamic(4)", omp_sched_dynamic, 4},
            {"guided(2)",  omp_sched_guided,  2},
        };

        for (int i = 0; i < 3; i++) {
            omp_set_schedule(scheds[i].kind, scheds[i].chunk);
            memset(result, 0, (size_t)ny * ny * sizeof(float));
            double t2 = omp_get_wtime();
            correlate_runtime(ny, nx, data, result);
            double par_time = omp_get_wtime() - t2;
            printf("%-12s | %-14.6f | %-14.6f | %-10.2f\n",
                   scheds[i].name, seq_time, par_time, seq_time / par_time);
        }

        free(data); free(result);
    }

    /* ---- Part 3: Performance Stats ---- */
    printf("\n=== Part 3: Performance Stats (ny=800, nx=5000, 4 threads, static) ===\n\n");
    {
        int ny = 800, nx = 5000;
        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(4);
        omp_set_schedule(omp_sched_static, 0);
        memset(result, 0, (size_t)ny * ny * sizeof(float));

        struct rusage ru0, ru1;
        getrusage(RUSAGE_SELF, &ru0);
        double wt0 = omp_get_wtime();
        correlate_runtime(ny, nx, data, result);
        double wall_time = omp_get_wtime() - wt0;
        getrusage(RUSAGE_SELF, &ru1);

        double task_ms = (ru1.ru_utime.tv_sec  - ru0.ru_utime.tv_sec ) * 1000.0
                       + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1000.0
                       + (ru1.ru_stime.tv_sec  - ru0.ru_stime.tv_sec ) * 1000.0
                       + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1000.0;

        double total_mb = (double)ny * nx * (sizeof(float) + sizeof(double)) / (1<<20)
                        + (double)ny * ny * sizeof(float) / (1<<20);

        printf("%-25s | %-15s\n", "Metric", "Value");
        printf("--------------------------|----------------\n");
        printf("%-25s | %.2f ms\n",  "Task Clock",              task_ms);
        printf("%-25s | %.2f\n",     "CPUs Utilized",           (task_ms / 1000.0) / wall_time);
        printf("%-25s | %ld\n",      "Context Switches (vol)",  ru1.ru_nvcsw  - ru0.ru_nvcsw);
        printf("%-25s | %ld\n",      "Context Switches (invol)",ru1.ru_nivcsw - ru0.ru_nivcsw);
        printf("%-25s | %ld\n",      "Page Faults (minor)",     ru1.ru_minflt - ru0.ru_minflt);
        printf("%-25s | %ld\n",      "Page Faults (major)",     ru1.ru_majflt - ru0.ru_majflt);
        printf("%-25s | %.6f s\n",   "Wall Clock Time",         wall_time);
        printf("%-25s | %.2f MB\n",  "Memory Footprint",        total_mb);

        free(data); free(result);
    }

    /* ---- Scheduling Policy Summary ---- */
    printf("\n=== Scheduling Policy Summary ===\n\n");
    printf("%-12s | %-30s | %-12s | %-30s\n",
           "Policy", "Work Distribution", "Overhead", "Observed Behaviour");
    printf("-------------|--------------------------------|--------------|-------------------------------\n");
    printf("%-12s | %-30s | %-12s | %-30s\n",
           "Static",     "Equal rows per thread",      "Lowest", "Imbalanced (triangular work)");
    printf("%-12s | %-30s | %-12s | %-30s\n",
           "Dynamic(4)", "Chunks of 4 rows on demand", "Higher", "Best for load balancing");
    printf("%-12s | %-30s | %-12s | %-30s\n",
           "Guided(2)",  "Decreasing chunk sizes",     "Medium", "Good balance, low overhead");

    return 0;
}

/* ================================================================== */
/*  QUES3 — Optimized vs Basic Parallel + Thread Scaling + Stats      */
/* ================================================================== */
#elif defined(QUES3)

int main()
{
    printf("Physical Cores: 10\n");
    printf("Logical Cores : 12 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    /* ---- Part 1: Optimized vs Basic Parallel ---- */
    printf("=== Part 1: Optimized vs Basic Parallel (4 threads) ===\n\n");
    printf("%-8s | %-8s | %-14s | %-14s | %-14s | %-10s\n",
           "ny", "nx", "Basic Par (s)", "Optimized (s)", "Seq Time (s)", "Opt Speedup");
    printf("---------|----------|----------------|----------------|----------------|----------\n");

    int test_ny[] = {300, 600, 1000};
    int test_nx[] = {3000, 5000, 8000};

    for (int s = 0; s < 3; s++) {
        int ny = test_ny[s], nx = test_nx[s];

        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(1);
        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t0 = omp_get_wtime();
        correlate_seq(ny, nx, data, result);
        double seq_time = omp_get_wtime() - t0;

        omp_set_num_threads(4);
        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t1 = omp_get_wtime();
        correlate_basic(ny, nx, data, result);
        double basic_time = omp_get_wtime() - t1;

        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t2 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double opt_time = omp_get_wtime() - t2;

        printf("%-8d | %-8d | %-14.6f | %-14.6f | %-14.6f | %-10.2f\n",
               ny, nx, basic_time, opt_time, seq_time, seq_time / opt_time);

        free(data); free(result);
    }

    /* ---- Part 2: Thread Scaling – Optimized ---- */
    printf("\n=== Part 2: Thread Scaling - Optimized (ny=800, nx=5000) ===\n\n");
    printf("%-8s | %-14s | %-14s | %-10s\n",
           "Threads", "Seq Time (s)", "Opt Time (s)", "Speedup");
    printf("---------|----------------|----------------|----------\n");
    {
        int ny = 800, nx = 5000;
        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(1);
        memset(result, 0, (size_t)ny * ny * sizeof(float));
        double t0 = omp_get_wtime();
        correlate_seq(ny, nx, data, result);
        double seq_time = omp_get_wtime() - t0;

        for (int t = 1; t <= 10; t++) {
            omp_set_num_threads(t);
            memset(result, 0, (size_t)ny * ny * sizeof(float));
            double t1 = omp_get_wtime();
            correlate(ny, nx, data, result);
            double opt_time = omp_get_wtime() - t1;
            printf("%-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   t, seq_time, opt_time, seq_time / opt_time);
        }

        free(data); free(result);
    }

    /* ---- Part 3: Performance Stats ---- */
    printf("\n=== Part 3: Performance Stats (ny=800, nx=5000, 4 threads, optimized) ===\n\n");
    {
        int ny = 800, nx = 5000;
        int padded_nx = (nx + 7) & ~7;
        float *data   = (float *)malloc((size_t)ny * nx * sizeof(float));
        float *result = (float *)malloc((size_t)ny * ny * sizeof(float));
        generate_data(data, ny, nx);

        omp_set_num_threads(4);
        memset(result, 0, (size_t)ny * ny * sizeof(float));

        struct rusage ru0, ru1;
        getrusage(RUSAGE_SELF, &ru0);
        double wt0 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double wall_time = omp_get_wtime() - wt0;
        getrusage(RUSAGE_SELF, &ru1);

        double task_ms = (ru1.ru_utime.tv_sec  - ru0.ru_utime.tv_sec ) * 1000.0
                       + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1000.0
                       + (ru1.ru_stime.tv_sec  - ru0.ru_stime.tv_sec ) * 1000.0
                       + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1000.0;

        double total_mb = (double)ny * nx        * sizeof(float)  / (1<<20)
                        + (double)ny * ny        * sizeof(float)  / (1<<20)
                        + (double)ny * padded_nx * sizeof(double) / (1<<20);

        double max_err = 0.0;
        for (int i = 0; i < ny; i++) {
            double e = fabs((double)result[i + i * ny] - 1.0);
            if (e > max_err) max_err = e;
        }

        printf("%-25s | %-15s\n", "Metric", "Value");
        printf("--------------------------|----------------\n");
        printf("%-25s | %.2f ms\n",  "Task Clock",              task_ms);
        printf("%-25s | %.2f\n",     "CPUs Utilized",           (task_ms / 1000.0) / wall_time);
        printf("%-25s | %ld\n",      "Context Switches (vol)",  ru1.ru_nvcsw  - ru0.ru_nvcsw);
        printf("%-25s | %ld\n",      "Context Switches (invol)",ru1.ru_nivcsw - ru0.ru_nivcsw);
        printf("%-25s | %ld\n",      "Page Faults (minor)",     ru1.ru_minflt - ru0.ru_minflt);
        printf("%-25s | %ld\n",      "Page Faults (major)",     ru1.ru_majflt - ru0.ru_majflt);
        printf("%-25s | %.6f s\n",   "Wall Clock Time",         wall_time);
        printf("%-25s | %.2f MB\n",  "Memory Footprint",        total_mb);
        printf("%-25s | %.2e\n",     "Max Diagonal Error",      max_err);

        free(data); free(result);
    }

    /* ---- Optimization Summary ---- */
    printf("\n=== Optimization Summary ===\n\n");
    printf("%-25s | %-35s | %-25s\n", "Optimization", "Description", "Effect");
    printf("--------------------------|-------------------------------------|-------------------------\n");
    printf("%-25s | %-35s | %-25s\n",
           "Memory Alignment",       "64-byte aligned, padded rows",     "Better SIMD vectorization");
    printf("%-25s | %-35s | %-25s\n",
           "ILP (4-way unroll)",      "4 dot products per iteration",    "Keeps FP units busy");
    printf("%-25s | %-35s | %-25s\n",
           "SIMD (#pragma omp simd)", "Compiler auto-vectorization hint","Uses AVX/SSE registers");
    printf("%-25s | %-35s | %-25s\n",
           "Dynamic Scheduling",      "Load-balanced triangular loop",   "Better thread utilization");
    printf("%-25s | %-35s | %-25s\n",
           "Row-major normalization", "Cache-friendly row access",        "Fewer cache misses");

    return 0;
}

#else
#error "Define QUES1, QUES2, or QUES3 when compiling."
#endif
