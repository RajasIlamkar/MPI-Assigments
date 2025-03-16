#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10  // Grid size (NxN)
#define MAX_ITER 1000  // Number of iterations
#define EPSILON 0.0001  // Convergence threshold

void initialize_grid(double grid[N][N]) {
    // Initialize grid with boundary conditions
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || j == 0 || i == N - 1 || j == N - 1)
                grid[i][j] = 100.0;  // Fixed boundary temperature
            else
                grid[i][j] = 0.0;  // Interior starts at 0
        }
    }
}

void print_grid(double grid[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", grid[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double grid[N][N], new_grid[N][N];
    int local_start, local_end;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    local_start = rank * rows_per_proc;
    local_end = (rank == size - 1) ? N : local_start + rows_per_proc;

    if (rank == 0) {
        initialize_grid(grid);
        printf("Initial Grid:\n");
        print_grid(grid);
    }

    MPI_Bcast(grid, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = local_start + 1; i < local_end - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
            }
        }

        // Exchange boundary rows with neighbors
        if (rank > 0)
            MPI_Sendrecv(&new_grid[local_start][0], N, MPI_DOUBLE, rank - 1, 0,
                         &new_grid[local_start - 1][0], N, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank < size - 1)
            MPI_Sendrecv(&new_grid[local_end - 1][0], N, MPI_DOUBLE, rank + 1, 0,
                         &new_grid[local_end][0], N, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy new grid values back
        for (int i = local_start; i < local_end; i++)
            for (int j = 0; j < N; j++)
                grid[i][j] = new_grid[i][j];

        MPI_Barrier(MPI_COMM_WORLD); // Sync all processes
    }

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nFinal Grid After %d Iterations:\n", MAX_ITER);
        print_grid(grid);
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
