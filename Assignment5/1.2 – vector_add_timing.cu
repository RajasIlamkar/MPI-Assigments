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
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cudaMemcpyToSymbol(A, h_A, sizeof(int) * N);
    cudaMemcpyToSymbol(B, h_B, sizeof(int) * N);

    // Setup CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<1, N>>>();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpyFromSymbol(h_C, C, sizeof(int) * N);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
