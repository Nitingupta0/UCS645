#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000          // Grid size (NxN)
#define STEPS 500       // Number of time steps
#define MAX_THREADS 10

double current[N][N];
double next[N][N];

/* Initialize temperature grid */
void initialize() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            current[i][j] = 0.0;
        }
    }

    /* Hot spot in the center */
    current[N/2][N/2] = 100.0;
}

/* Heat diffusion simulation */
double heat_diffusion(int threads, double *total_heat) {

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    for (int t = 0; t < STEPS; t++) {

        double heat_sum = 0.0;

        #pragma omp parallel for schedule(runtime) reduction(+:heat_sum)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i][j] = 0.25 * (
                    current[i-1][j] +
                    current[i+1][j] +
                    current[i][j-1] +
                    current[i][j+1]
                );
                heat_sum += next[i][j];
            }
        }

        /* Swap grids */
        #pragma omp parallel for schedule(runtime)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                current[i][j] = next[i][j];
            }
        }

        *total_heat = heat_sum;
    }

    double end = omp_get_wtime();
    return end - start;
}

int main() {

    double serial_time, time_taken, speedup;
    double total_heat;

    printf("\nHeat Diffusion Simulation (2D)\n");
    printf("Grid size        : %d x %d\n", N, N);
    printf("Time steps       : %d\n", STEPS);
    printf("Scheduling       : OMP_SCHEDULE (runtime)\n\n");

    printf("Threads\tTime (s)\tSpeedup\n");
    printf("---------------------------------\n");

    /* Baseline: 1 thread */
    initialize();
    serial_time = heat_diffusion(1, &total_heat);
    printf("1\t%f\t1.00\n", serial_time);

    /* Parallel runs */
    for (int t = 2; t <= MAX_THREADS; t++) {
        initialize();
        time_taken = heat_diffusion(t, &total_heat);
        speedup = serial_time / time_taken;
        printf("%d\t%f\t%.2f\n", t, time_taken, speedup);
    }

    printf("\nFinal total heat : %f\n", total_heat);

    return 0;
}
