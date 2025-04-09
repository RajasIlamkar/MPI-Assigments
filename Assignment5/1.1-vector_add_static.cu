#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__device__ int A[N];
__device__ int B[N];
__device__ int C[N];

__global__ void vectorAdd() {
    int idx = threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int h_A[N], h_B[N], h_C[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Copy to device statically defined global arrays
    cudaMemcpyToSymbol(A, h_A, sizeof(int) * N);
    cudaMemcpyToSymbol(B, h_B, sizeof(int) * N);

    // Launch kernel
    vectorAdd<<<1, N>>>();
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpyFromSymbol(h_C, C, sizeof(int) * N);

    // Verify results
    for (int i = 0; i < 5; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    return 0;
}
