#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 16)  // Size of vectors

void daxpy(int n, double a, double *X, double *Y) {
    for (int i = 0; i < n; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *X = NULL, *Y = NULL;
    double *local_X, *local_Y;
    double a = 2.5;
    int local_n;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;  // Work per process

    // Allocate memory for local data
    local_X = (double *)malloc(local_n * sizeof(double));
    local_Y = (double *)malloc(local_n * sizeof(double));

    // Root initializes full vectors
    if (rank == 0) {
        X = (double *)malloc(N * sizeof(double));
        Y = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            X[i] = 1.0;  // Example values
            Y[i] = 2.0;
        }
    }

    // Scatter data among processes
    MPI_Scatter(X, local_n, MPI_DOUBLE, local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, local_n, MPI_DOUBLE, local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timer
    start_time = MPI_Wtime();

    // Perform DAXPY operation on local data
    daxpy(local_n, a, local_X, local_Y);

    // End timer
    end_time = MPI_Wtime();

    // Gather results back to root process
    MPI_Gather(local_X, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root prints the execution time
    if (rank == 0) {
        printf("MPI Execution Time: %f seconds\n", end_time - start_time);
        free(X);
        free(Y);
    }

    // Free local memory
    free(local_X);
    free(local_Y);

    MPI_Finalize();
    return 0;
}
