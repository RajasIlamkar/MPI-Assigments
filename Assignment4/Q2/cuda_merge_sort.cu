#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1000

// CUDA merge kernel (each thread block merges a pair of subarrays)
__global__ void mergePass(int* input, int* output, int width, int n) {
    int idx = blockIdx.x;

    int left = idx * 2 * width;
    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);

    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        output[k++] = (input[i] < input[j]) ? input[i++] : input[j++];
    }

    while (i < mid) output[k++] = input[i++];
    while (j < right) output[k++] = input[j++];
}

// Host function to call merge sort
void cudaMergeSort(int* h_data) {
    int *d_input, *d_output;
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice);

    int* src = d_input;
    int* dst = d_output;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Bottom-up merge sort: increase width each pass
    for (int width = 1; width < N; width *= 2) {
        int numBlocks = (N + 2 * width - 1) / (2 * width);
        mergePass<<<numBlocks, 1>>>(src, dst, width, N);
        cudaDeviceSynchronize();

        // Swap buffers
        int* temp = src;
        src = dst;
        dst = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Merge Sort Time: " << milliseconds << " ms" << std::endl;

    // Copy sorted array back
    cudaMemcpy(h_data, src, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int* h_data = new int[N];
    for (int i = 0; i < N; ++i)
        h_data[i] = rand() % 10000;

    cudaMergeSort(h_data);

    // Optional: check result
    // for (int i = 0; i < N; ++i)
    //     std::cout << h_data[i] << " ";

    delete[] h_data;
    return 0;
}
