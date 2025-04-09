#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of integers

// CUDA Kernel Function
__global__ void computeSums(int* input, int* output) {
    int tid = threadIdx.x;

    // Task a: Iterative sum by thread 0
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += input[i];
        }
        output[0] = sum;
    }

    // Task b: Formula sum by thread 1
    if (tid == 1) {
        int sum = (N * (N - 1)) / 2;
        output[1] = sum;
    }
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;

    size_t size = N * sizeof(int);
    size_t outSize = 2 * sizeof(int);  // Only 2 outputs

    // Allocate memory on host
    h_input = (int*)malloc(size);
    h_output = (int*)malloc(outSize);

    // Fill input array with first N integers: 0 to N-1
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, outSize);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and at least 2 threads
    computeSums<<<1, 2>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, outSize, cudaMemcpyDeviceToHost);

    // Display results
    std::cout << "Sum using iterative approach: " << h_output[0] << std::endl;
    std::cout << "Sum using formula: " << h_output[1] << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
