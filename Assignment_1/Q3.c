#include <stdio.h>
#include <time.h>
#include <omp.h>

static long NUM_STEPS = 100000000;

int main() {

    double step = 1.0 / (double)NUM_STEPS;
    double sum, pi;
    double serial_time, parallel_time, speedup;

    /* ================= SERIAL (NO OpenMP) ================= */

    sum = 0.0;
    clock_t s_start = clock();

    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    clock_t s_end = clock();

    serial_time = (double)(s_end - s_start) / CLOCKS_PER_SEC;

    printf("\n===== PI CALCULATION PERFORMANCE =====\n");
    printf("Serial Pi Value : %.15f\n", pi);
    printf("Serial Time     : %f seconds\n\n", serial_time);

    /* ================= OpenMP ================= */

    printf("OpenMP Execution\n");
    printf("Threads\tTime (s)\tSpeedup\t\tPi Value\n");
    printf("--------------------------------------------------------------\n");

    for (int threads = 1; threads <= 16; threads++) {

        sum = 0.0;
        omp_set_num_threads(threads);

        double p_start = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (long i = 0; i < NUM_STEPS; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        pi = step * sum;
        double p_end = omp_get_wtime();

        parallel_time = p_end - p_start;
        speedup = serial_time / parallel_time;

        printf("%d\t%f\t%.2f\t\t%.15f\n",
               threads, parallel_time, speedup, pi);
    }

    return 0;
}
