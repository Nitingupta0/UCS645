#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000
#define EPSILON 1.0
#define SIGMA 1.0

double x[N][3];
double f[N][3];

/* Initialize particle positions */
void initialize_particles() {
    for (int i = 0; i < N; i++) {
        x[i][0] = (double)rand() / RAND_MAX;
        x[i][1] = (double)rand() / RAND_MAX;
        x[i][2] = (double)rand() / RAND_MAX;
    }
}

/* Clear force array */
void clear_forces() {
    for (int i = 0; i < N; i++)
        f[i][0] = f[i][1] = f[i][2] = 0.0;
}

/* Optimized force calculation */
double compute_forces(int threads) {

    double potential_energy = 0.0;

    omp_set_num_threads(threads);

    /* Thread-private force buffers */
    double (*f_private)[N][3] =
        malloc(sizeof(double) * threads * N * 3);

    for (int t = 0; t < threads; t++)
        for (int i = 0; i < N; i++)
            f_private[t][i][0] =
            f_private[t][i][1] =
            f_private[t][i][2] = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel reduction(+:potential_energy)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {

                double dx = x[i][0] - x[j][0];
                double dy = x[i][1] - x[j][1];
                double dz = x[i][2] - x[j][2];

                double r2 = dx*dx + dy*dy + dz*dz;
                double inv_r2 = 1.0 / r2;
                double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                double inv_r12 = inv_r6 * inv_r6;

                double vij = 4.0 * EPSILON * (inv_r12 - inv_r6);
                potential_energy += vij;

                double fij = 24.0 * EPSILON *
                             (2.0 * inv_r12 - inv_r6) * inv_r2;

                double fx = fij * dx;
                double fy = fij * dy;
                double fz = fij * dz;

                f_private[tid][i][0] += fx;
                f_private[tid][i][1] += fy;
                f_private[tid][i][2] += fz;

                f_private[tid][j][0] -= fx;
                f_private[tid][j][1] -= fy;
                f_private[tid][j][2] -= fz;
            }
        }
    }

    /* Combine forces */
    for (int t = 0; t < threads; t++)
        for (int i = 0; i < N; i++) {
            f[i][0] += f_private[t][i][0];
            f[i][1] += f_private[t][i][1];
            f[i][2] += f_private[t][i][2];
        }

    double end = omp_get_wtime();
    free(f_private);

    return end - start;
}

int main() {

    double serial_time, parallel_time, speedup;

    initialize_particles();

    printf("Threads\tTime (s)\tSpeedup\n");
    printf("---------------------------------\n");

    clear_forces();
    serial_time = compute_forces(1);
    printf("1\t%f\t1.00\n", serial_time);

    for (int threads = 2; threads <= 16; threads++) {
        clear_forces();
        parallel_time = compute_forces(threads);
        speedup = serial_time / parallel_time;

        printf("%d\t%f\t%.2f\n",
               threads, parallel_time, speedup);
    }

    return 0;
}
