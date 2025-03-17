#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 100  // Upper limit for primes

// Function to check if a number is prime
int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size, num;
    int prime_list[MAX_VALUE];  // Store primes
    int prime_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {  // Master Process
        int next_number = 2;  // Start checking from 2
        int received_number, sender_rank;
        MPI_Status status;

        while (next_number <= MAX_VALUE || size > 1) {
            // Receive message from any worker
            MPI_Recv(&received_number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sender_rank = status.MPI_SOURCE;

            // Process the result
            if (received_number > 0) {  // Prime number found
                prime_list[prime_count++] = received_number;
            }

            // Send next number to test
            if (next_number <= MAX_VALUE) {
                MPI_Send(&next_number, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);
                next_number++;
            } else {  
                // No more numbers to test, send termination signal (-1)
                int terminate_signal = -1;
                MPI_Send(&terminate_signal, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);
                size--;  // Reduce active workers count
            }
        }

        // Print prime numbers
        printf("Primes up to %d: ", MAX_VALUE);
        for (int i = 0; i < prime_count; i++) {
            printf("%d ", prime_list[i]);
        }
        printf("\n");

    } else {  // Worker Processes
        int test_number;
        int result;
        
        // Send initial message (0) to indicate readiness
        num = 0;
        MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            // Receive a number to test from master
            MPI_Recv(&test_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // If termination signal received, exit loop
            if (test_number < 0) break;

            // Check if the number is prime
            if (is_prime(test_number)) {
                result = test_number;  // Positive if prime
            } else {
                result = -test_number;  // Negative if not prime
            }

            // Send result back to master
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
