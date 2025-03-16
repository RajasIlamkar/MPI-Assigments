#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 16)  // 2^16 = 65536 elements

// Function to initialize vectors
void initialize_vectors(double *X, double *Y, int size) {
    for (int i = 0; i < size; i++) {
        X[i] = 1.0;   // Example initialization
        Y[i] = 2.0;   // Example initialization
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double a = 3.5;  // Scalar multiplier
    double *X = NULL, *Y = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;  // Each process gets a portion

    // Allocate local portions of X and Y
    double *local_X = (double *)malloc(local_n * sizeof(double));
    double *local_Y = (double *)malloc(local_n * sizeof(double));

    // Master process initializes the full vectors
    if (rank == 0) {
        X = (double *)malloc(N * sizeof(double));
        Y = (double *)malloc(N * sizeof(double));
        initialize_vectors(X, Y, N);
    }

    // Scatter the data to all processes
    MPI_Scatter(X, local_n, MPI_DOUBLE, local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, local_n, MPI_DOUBLE, local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timer
    start_time = MPI_Wtime();

    // Perform local DAXPY computation
    for (int i = 0; i < local_n; i++) {
        local_X[i] = a * local_X[i] + local_Y[i];
    }

    // End timer
    end_time = MPI_Wtime();

    // Gather the results back
    MPI_Gather(local_X, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print execution time on master
    if (rank == 0) {
        printf("Parallel Execution Time: %f seconds\n", end_time - start_time);
        free(X);
        free(Y);
    }

    // Free allocated memory
    free(local_X);
    free(local_Y);

    MPI_Finalize();
    return 0;
}
