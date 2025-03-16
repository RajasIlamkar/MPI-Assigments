#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOTAL_POINTS 1000000  // Total number of points for the simulation

int main(int argc, char** argv) {
    int rank, size, local_count = 0, global_count;
    int points_per_process;
    double x, y, pi_estimate;
    
    MPI_Init(&argc, &argv);                 // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get total number of processes

    points_per_process = TOTAL_POINTS / size;  // Divide work among processes
    srand(time(NULL) + rank);  // Seed the random number generator

    // Monte Carlo Simulation
    for (int i = 0; i < points_per_process; i++) {
        x = (double)rand() / RAND_MAX;  // Random x in [0,1]
        y = (double)rand() / RAND_MAX;  // Random y in [0,1]

        if (x * x + y * y <= 1) {
            local_count++;  // Point is inside the circle
        }
    }

    // Gather results from all processes
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Final computation by rank 0
    if (rank == 0) {
        pi_estimate = (4.0 * global_count) / TOTAL_POINTS;
        printf("Estimated Pi value: %f\n", pi_estimate);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
