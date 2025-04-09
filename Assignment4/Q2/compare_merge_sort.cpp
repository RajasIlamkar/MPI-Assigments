#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>

#define N 1000

// ---------- CPU MERGE SORT ----------

void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid+1, k = 0;

    while (i <= mid && j <= right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];

    while (i <= mid)  temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < temp.size(); ++i)
        arr[left + i] = temp[i];
}

void parallelMergeSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int width = 1; width < n; width *= 2) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i += 2 * width) {
            int left = i;
            int mid = std::min(i + width - 1, n - 1);
            int right = std::min(i + 2 * width - 1, n - 1);
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}

// ---------- CUDA MERGE SORT ----------

__global__ void mergePass(int* input, int* output, int width, int n) {
    int idx = blockIdx.x;
    int left = idx * 2 * width;
    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);

    int i = left, j = mid, k = left;

    while (i < mid && j < right)
        output[k++] = (input[i] < input[j]) ? input[i++] : input[j++];
    while (i < mid) output[k++] = input[i++];
    while (j < right) output[k++] = input[j++];
}

float cudaMergeSort(std::vector<int>& host_data) {
    int* d_input;
    int* d_output;
    int* h_data = host_data.data();
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice);

    int* src = d_input;
    int* dst = d_output;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int width = 1; width < N; width *= 2) {
        int numBlocks = (N + 2 * width - 1) / (2 * width);
        mergePass<<<numBlocks, 1>>>(src, dst, width, N);
        cudaDeviceSynchronize();
        std::swap(src, dst);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_data, src, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// ---------- MAIN ----------

bool isSorted(const std::vector<int>& arr) {
    for (int i = 1; i < arr.size(); ++i)
        if (arr[i - 1] > arr[i]) return false;
    return true;
}

int main() {
    std::vector<int> original(N);
    for (int i = 0; i < N; ++i)
        original[i] = rand() % 10000;

    std::vector<int> cpu_data = original;
    std::vector<int> gpu_data = original;

    // CPU timing
    auto start = std::chrono::high_resolution_clock::now();
    parallelMergeSort(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end - start;

    // GPU timing
    float gpu_time = cudaMergeSort(gpu_data);

    // Output timings
    std::cout << "CPU Merge Sort Time (OpenMP): " << cpu_time.count() << " ms" << std::endl;
    std::cout << "CUDA Merge Sort Time: " << gpu_time << " ms" << std::endl;

    // Optional: compare results
    bool same = (cpu_data == gpu_data);
    std::cout << "Arrays Match: " << (same ? "YES" : "NO") << std::endl;
    std::cout << "CPU Sorted: " << (isSorted(cpu_data) ? "YES" : "NO") << std::endl;
    std::cout << "GPU Sorted: " << (isSorted(gpu_data) ? "YES" : "NO") << std::endl;

    return 0;
}
