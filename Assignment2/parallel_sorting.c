#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 12  // Size of array

// Function to perform local bubble sort
void local_bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to swap boundary elements with neighbors
void swap(int *a, int *b) {
    if (*a > *b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *global_array = NULL;
    int local_array[N];
    int local_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_size = N / size;

    // Root process initializes the array
    if (rank == 0) {
        global_array = (int *)malloc(N * sizeof(int));
        srand(time(NULL));
        printf("Unsorted Array: ");
        for (int i = 0; i < N; i++) {
            global_array[i] = rand() % 100;
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }

    // Scatter the array to all processes
    MPI_Scatter(global_array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local sorting
    local_bubble_sort(local_array, local_size);

    for (int phase = 0; phase < size; phase++) {
        int partner;
        
        if (phase % 2 == 0) {  // Even phase
            if (rank % 2 == 0)
                partner = rank + 1;
            else
                partner = rank - 1;
        } else {  // Odd phase
            if (rank % 2 == 0)
                partner = rank - 1;
            else
                partner = rank + 1;
        }

        if (partner >= 0 && partner < size) {
            int boundary_element, recv_element;
            
            if (rank < partner) {
                boundary_element = local_array[local_size - 1];
                MPI_Send(&boundary_element, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                MPI_Recv(&recv_element, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_array[local_size - 1] = recv_element;
            } else {
                MPI_Recv(&recv_element, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                boundary_element = local_array[0];
                MPI_Send(&boundary_element, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                local_array[0] = recv_element;
            }
        }
    }

    // Gather the sorted sections back to the root process
    MPI_Gather(local_array, local_size, MPI_INT, global_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sorted Array: ");
        for (int i = 0; i < N; i++)
            printf("%d ", global_array[i]);
        printf("\n");

        free(global_array);
    }

    MPI_Finalize();
    return 0;
}
