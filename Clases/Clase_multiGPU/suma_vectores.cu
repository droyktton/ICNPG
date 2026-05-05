#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Standard Vector Addition Kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 1) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " GPU(s). Splitting workload..." << std::endl;

    const int N = 1 << 20; // Total elements
    const int elementsPerGPU = N / deviceCount;
    const size_t sizePerGPU = elementsPerGPU * sizeof(float);

    // Host data
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    // Pointers for each GPU
    std::vector<float*> d_A(deviceCount), d_B(deviceCount), d_C(deviceCount);

    // 1. Launch / Data Transfer Phase
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i); // Switch to GPU i

        // Allocate memory on current GPU
        cudaMalloc(&d_A[i], sizePerGPU);
        cudaMalloc(&d_B[i], sizePerGPU);
        cudaMalloc(&d_C[i], sizePerGPU);

        // Copy subset of host data to current GPU
        int offset = i * elementsPerGPU;
        cudaMemcpy(d_A[i], &h_A[offset], sizePerGPU, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B[i], &h_B[offset], sizePerGPU, cudaMemcpyHostToDevice);

        // Launch kernel on current GPU
        int threadsPerBlock = 256;
        int blocksPerGrid = (elementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], elementsPerGPU);
    }

    // 2. Synchronization and Retrieval Phase
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        
        // Wait for this GPU to finish (Kernel launches are asynchronous to host)
        cudaDeviceSynchronize();

        // Copy result back to the correct offset in host memory
        int offset = i * elementsPerGPU;
        cudaMemcpy(&h_C[offset], d_C[i], sizePerGPU, cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }

    // Verify
    if (h_C[0] == 3.0f && h_C[N-1] == 3.0f) {
        std::cout << "Success! Multi-GPU computation complete." << std::endl;
    }

    return 0;
}
