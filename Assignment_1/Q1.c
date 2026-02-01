#include <stdio.h>
#include <time.h>
#include <omp.h>

#define N (1 << 24)

double X[N], Y[N];

int main() {
    double a = 2.5;
    int i;

    double serial_time;
    double parallel_time;
    double speedup;

    /* ================= SERIAL (NO OpenMP) ================= */

    for (i = 0; i < N; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    clock_t s_start = clock();

    for (i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    clock_t s_end = clock();

    serial_time = (double)(s_end - s_start) / CLOCKS_PER_SEC;

    printf("\n===== DAXPY Performance =====\n");
    printf("Serial Execution Time: %f seconds\n\n", serial_time);

    /* ================= OpenMP ================= */

    printf("OpenMP Execution\n");
    printf("Threads\tTime (s)\tSpeedup\n");
    printf("----------------------------------\n");

    for (int threads = 1; threads <= 16; threads++) {

        /* Reset arrays for fair comparison */
        for (i = 0; i < N; i++) {
            X[i] = i * 1.0;
            Y[i] = i * 2.0;
        }

        omp_set_num_threads(threads);

        double p_start = omp_get_wtime();

        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            X[i] = a * X[i] + Y[i];
        }

        double p_end = omp_get_wtime();

        parallel_time = p_end - p_start;
        speedup = serial_time / parallel_time;

        printf("%d\t%f\t%.2f\n", threads, parallel_time, speedup);
    }

    return 0;
}
