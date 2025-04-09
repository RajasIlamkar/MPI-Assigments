#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int memClockKHz = prop.memoryClockRate;  // in kHz
    int memBusWidth = prop.memoryBusWidth;   // in bits

    float theoreticalBandwidth = 2.0f * memClockKHz * 1000 * (memBusWidth / 8.0f); // in Bytes/sec
    theoreticalBandwidth /= 1e9f; // Convert to GB/s

    std::cout << "Memory Clock Rate (kHz): " << memClockKHz << std::endl;
    std::cout << "Memory Bus Width (bits): " << memBusWidth << std::endl;
    std::cout << "Theoretical Bandwidth: " << theoreticalBandwidth << " GB/s" << std::endl;

    return 0;
}
