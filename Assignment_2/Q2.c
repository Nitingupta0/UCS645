#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -1

#define LEN 1000

int max(int a, int b) {
    return (a > b) ? a : b;
}

int max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

/* Generate random DNA sequence */
void generate_sequence(char *seq, int len) {
    char dna[] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < len; i++)
        seq[i] = dna[rand() % 4];
    seq[len] = '\0';
}

/* Smith–Waterman computation (single run) */
double smith_waterman(int threads, int *final_score) {

    static int H[LEN + 1][LEN + 1];
    char seqA[LEN + 1], seqB[LEN + 1];

    generate_sequence(seqA, LEN);
    generate_sequence(seqB, LEN);

    for (int i = 0; i <= LEN; i++)
        for (int j = 0; j <= LEN; j++)
            H[i][j] = 0;

    int max_score = 0;
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    /* Wavefront parallelization */
    for (int d = 1; d <= 2 * LEN; d++) {

        #pragma omp parallel for schedule(runtime) reduction(max:max_score)
        for (int i = 1; i <= LEN; i++) {

            int j = d - i;
            if (j >= 1 && j <= LEN) {

                int score_diag = H[i-1][j-1] +
                    (seqA[i-1] == seqB[j-1] ? MATCH : MISMATCH);

                int score_up   = H[i-1][j] + GAP;
                int score_left = H[i][j-1] + GAP;

                H[i][j] = max4(0, score_diag, score_up, score_left);

                if (H[i][j] > max_score)
                    max_score = H[i][j];
            }
        }
    }

    double end = omp_get_wtime();

    *final_score = max_score;
    return end - start;
}

int main() {

    double serial_time = 0.0;
    int score;

    printf("\nSmith–Waterman Local Alignment\n");
    printf("Sequence length : %d\n", LEN);
    printf("Scheduling      : OMP_SCHEDULE (runtime)\n\n");

    printf("Threads\tTime (s)\tSpeedup\n");
    printf("--------------------------------------\n");

    /* Baseline: 1 thread */
    serial_time = smith_waterman(1, &score);
    printf("1\t%f\t1.00\n", serial_time);

    /* Parallel runs */
    for (int t = 2; t <= 8; t++) {
        double time = smith_waterman(t, &score);
        double speedup = serial_time / time;
        printf("%d\t%f\t%.2f\n", t, time, speedup);
    }

    printf("\nAlignment score (all runs): %d\n", score);

    return 0;
}
