#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 1000  // Define the total size of the vectors

// Function to compute the local dot product
double local_dot_product(double *a, double *b, int size) {
    double local_sum = 0.0;
    for (int i = 0; i < size; i++) {
        local_sum += a[i] * b[i];
    }
    return local_sum;
}

int main(int argc, char **argv) {
    int rank, size;
    int n = VECTOR_SIZE;
    double *A = NULL, *B = NULL;
    double local_sum = 0.0, global_sum = 0.0;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Determine local size
    int local_n = n / size;
    
    // Allocate local arrays
    double *local_A = (double *)malloc(local_n * sizeof(double));
    double *local_B = (double *)malloc(local_n * sizeof(double));

    // Master process initializes the vectors
    if (rank == 0) {
        A = (double *)malloc(n * sizeof(double));
        B = (double *)malloc(n * sizeof(double));
        
        for (int i = 0; i < n; i++) {
            A[i] = 1.0;  // Example values
            B[i] = 1.0;  // Example values
        }
    }

    // Scatter the vectors to all processes
    MPI_Scatter(A, local_n, MPI_DOUBLE, local_A, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_DOUBLE, local_B, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local dot product
    local_sum = local_dot_product(local_A, local_B, local_n);

    // Reduce the local dot products to get the global dot product
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Master process prints the final result
    if (rank == 0) {
        printf("Dot Product: %f\n", global_sum);
    }

    // Free memory
    free(local_A);
    free(local_B);
    if (rank == 0) {
        free(A);
        free(B);
    }

    MPI_Finalize();
    return 0;
}
