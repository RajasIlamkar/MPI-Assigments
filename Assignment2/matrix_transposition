#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4  // Matrix size (NxN)

void print_matrix(int *matrix, int n, const char *msg) {
    printf("%s\n", msg);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int matrix[N][N], transposed[N][N];
    int local_matrix[N][N], local_transposed[N][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure perfect division
    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size must be divisible by number of processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    int rows_per_proc = N / size;

    // Initialize matrix in rank 0
    if (rank == 0) {
        int counter = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = counter++;
            }
        }
        print_matrix((int *)matrix, N, "Original Matrix:");
    }

    // Scatter matrix rows to all processes
    MPI_Scatter(matrix, rows_per_proc * N, MPI_INT, local_matrix, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Local transposition (swap rows and columns)
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_transposed[j][i + rank * rows_per_proc] = local_matrix[i][j];
        }
    }

    // Gather transposed parts
    MPI_Gather(local_transposed, rows_per_proc * N, MPI_INT, transposed, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Print result in rank 0
    if (rank == 0) {
        print_matrix((int *)transposed, N, "Transposed Matrix:");
    }

    MPI_Finalize();
    return 0;
}
