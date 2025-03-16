#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 70  // Matrix size

void multiply_matrices(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }

    // Measure execution time
    double start_time = omp_get_wtime();
    multiply_matrices(A, B, C);
    double run_time = omp_get_wtime() - start_time;

    printf("Serial Execution Time: %f seconds\n", run_time);
    return 0;
}
