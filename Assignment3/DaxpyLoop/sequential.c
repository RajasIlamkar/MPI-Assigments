#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 16)  // 65536 elements

int main() {
    double a = 3.5;
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double start_time = omp_get_wtime();
    
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
    
    double end_time = omp_get_wtime();
    printf("Sequential Execution Time: %f seconds\n", end_time - start_time);
    
    free(X);
    free(Y);
    return 0;
}
