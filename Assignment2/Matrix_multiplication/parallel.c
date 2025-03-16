#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 70  // Matrix size

void multiply_matrices(double A[N][N], double B[N][N], double C[N][N], int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A[N][N], B[N][N], C[N][N];
    int rows_per_process = N / size;
    int start = rank * rows_per_process;
    int end = (rank == size - 1) ? N : start + rows_per_process;

    // Initialize matrices
    if (rank == 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
                C[i][j] = 0;
            }
    }

    // Broadcast matrices A and B to all processes
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure parallel execution time
    double start_time = omp_get_wtime();

    // Perform partial matrix multiplication
    multiply_matrices(A, B, C, start, end);

    // Gather the results at root process
    MPI_Gather(C[start], rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double run_time = omp_get_wtime() - start_time;

    // Display execution time at root
    if (rank == 0) {
        printf("MPI Parallel Execution Time: %f seconds\n", run_time);
    }

    MPI_Finalize();
    return 0;
}
