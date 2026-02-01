#include <stdio.h>
#include <time.h>
#include <omp.h>

#define N 1000   // Matrix size (NxN)

double A[N][N], B[N][N], C[N][N];

/* Initialize matrices */
void initialize() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
    }
}

/* Clear result matrix */
void clear_C() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0;
}

/* -------- SERIAL MATRIX MULTIPLICATION -------- */
double matmul_serial() {
    initialize();
    clear_C();

    double start = omp_get_wtime();

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    double end = omp_get_wtime();
    return (end - start);
}

/* -------- OPENMP 1D THREADING -------- */
double matmul_1D(int threads) {
    initialize();
    clear_C();
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    double end = omp_get_wtime();
    return (end - start);
}

/* -------- OPENMP 2D THREADING -------- */
double matmul_2D(int threads) {
    initialize();
    clear_C();
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    double end = omp_get_wtime();
    return (end - start);
}

int main() {

    printf("\n===== Matrix Multiplication Performance =====\n");

    /* -------- SERIAL BASELINE -------- */
    double serial_time = matmul_serial();
    printf("\nSerial Execution Time: %f seconds\n", serial_time);

    /* -------- OPENMP 1D THREADING -------- */
    printf("\n--- OpenMP 1D Threading ---\n");
    printf("Threads\tTime (s)\tSpeedup\n");
    printf("----------------------------------\n");

    for (int threads = 1; threads <= 16; threads++) {
        double time_taken = matmul_1D(threads);
        double speedup = serial_time / time_taken;
        printf("%d\t%f\t%.2f\n", threads, time_taken, speedup);
    }

    /* -------- OPENMP 2D THREADING -------- */
    printf("\n--- OpenMP 2D Threading ---\n");
    printf("Threads\tTime (s)\tSpeedup\n");
    printf("----------------------------------\n");

    for (int threads = 1; threads <= 16; threads++) {
        double time_taken = matmul_2D(threads);
        double speedup = serial_time / time_taken;
        printf("%d\t%f\t%.2f\n", threads, time_taken, speedup);
    }

    return 0;
}
