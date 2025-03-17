#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 16)

void daxpy(int n, double a, double *X, double *Y) {
    for (int i = 0; i < n; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main() {
    double *X, *Y;
    double a = 2.5;
    clock_t start, end;

    // Allocate memory
    X = (double *)malloc(N * sizeof(double));
    Y = (double *)malloc(N * sizeof(double));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    // Start timer
    start = clock();

    // Perform DAXPY operation
    daxpy(N, a, X, Y);

    // End timer
    end = clock();

    printf("Serial Execution Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Free memory
    free(X);
    free(Y);

    return 0;
}
