#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;

int main() {
    double step = 1.0 / (double) num_steps;
    double sum = 0.0, pi;
    int i;

    #pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    printf("Estimated Pi = %.15f\n", pi);

    return 0;
}
