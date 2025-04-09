#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#define N 1000  // Array size

// Merge function
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

// Bottom-up Merge Sort with OpenMP pipelining
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

int main() {
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = rand() % 10000;

    auto start = std::chrono::high_resolution_clock::now();
    parallelMergeSort(data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Pipelined CPU Merge Sort Time: " << duration.count() << " ms" << std::endl;

    // Optional: check sorted array
    // for (int i : data) std::cout << i << " ";
}
