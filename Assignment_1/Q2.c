#include <stdio.h>
#include <stdlib.h>
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


  // Version 1: 1D Threading

void matmul_1D() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


  // Version 2: 2D Threading

void matmul_2D() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double start, end;

    initialize();

    // 1D Threading
    clear_C();
    start = omp_get_wtime();
    matmul_1D();
    end = omp_get_wtime();
    printf("1D Threading Time: %f seconds\n", end - start);

    //2D Threading 
    clear_C();
    start = omp_get_wtime();
    matmul_2D();
    end = omp_get_wtime();
    printf("2D Threading Time: %f seconds\n", end - start);

    return 0;
}