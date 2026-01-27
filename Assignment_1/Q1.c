#include <stdio.h>
#include <omp.h>

#define N (1<<24)

int main() {
    double X[N], Y[N], a = 2.5;
    int i;

    for (i = 0; i < N; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();
    printf("Execution time: %f seconds\n", end - start);

    return 0;
}
