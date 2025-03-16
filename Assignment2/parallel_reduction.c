#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100  // Size of array

int main(int argc, char *argv[]) {
    int rank, size;
    int local_sum = 0, global_sum = 0;
    int *arr = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = N / size;
    int *local_arr = (int *)malloc(local_size * sizeof(int));

    if (rank == 0) {
        arr = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) arr[i] = i + 1;  // Fill with numbers 1 to N
    }

    // Scatter array to all processes
    MPI_Scatter(arr, local_size, MPI_INT, local_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local sum
    for (int i = 0; i < local_size; i++) local_sum += local_arr[i];

    // Reduce all local sums to get the global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Sum: %d\n", global_sum);
        free(arr);
    }

    free(local_arr);
    MPI_Finalize();
    return 0;
}
