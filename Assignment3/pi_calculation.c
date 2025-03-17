#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static long num_steps = 100000;  // Number of steps for approximation
double step;

int main(int argc, char *argv[]) {
    int rank, size, i;
    double x, sum = 0.0, local_sum = 0.0;
    double pi, start_time, end_time;

    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Calculating π using %d processes and %ld steps.\n", size, num_steps);
    }

    // Broadcast num_steps to all processes
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;  // Compute step size

    // Start timer
    start_time = MPI_Wtime();

    // Divide workload among processes
    for (i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Reduce all local sums into global sum at process 0
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Compute final pi value and print results
    if (rank == 0) {
        pi = step * sum;
        end_time = MPI_Wtime();
        printf("Calculated π: %.15f\n", pi);
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
