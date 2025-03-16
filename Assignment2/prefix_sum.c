#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8  // Number of elements in the array

int main(int argc, char** argv) {
    int rank, size;
    int local_value, prefix_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize data (each process gets one element for simplicity)
    local_value = rank + 1;  // Example: P0 gets 1, P1 gets 2, etc.

    // Perform parallel prefix sum (inclusive)
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Print results
    printf("Process %d: Local value = %d, Prefix sum = %d\n", rank, local_value, prefix_sum);

    MPI_Finalize();
    return 0;
}
