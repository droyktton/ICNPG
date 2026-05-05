#include <iostream>
#include <cuda_runtime.h>

int main() {
    int dev0 = 0, dev1 = 1;
    int canAccessPeer;

    // 1. Check if GPU 0 can access GPU 1
    cudaDeviceCanAccessPeer(&canAccessPeer, dev0, dev1);
    if (!canAccessPeer) {
        std::cout << "P2P access not supported between GPU 0 and 1" << std::endl;
        return 0;
    }

    // 2. Enable Peer Access (Unidirectional)
    cudaSetDevice(dev0);
    cudaDeviceEnablePeerAccess(dev1, 0); // Allow GPU 0 to see GPU 1 

    cudaSetDevice(dev1);
    cudaDeviceEnablePeerAccess(dev0, 0); // Allow GPU 1 to see GPU 0 

    // 3. Allocate memory on both GPUs
    size_t size = 1024 * sizeof(float);
    float *d_0, *d_1;

    cudaSetDevice(dev0);
    cudaMalloc(&d_0, size);

    cudaSetDevice(dev1);
    cudaMalloc(&d_1, size);

    // 4. Perform Direct Peer-to-Peer Copy
    // This moves data directly over NVLink/PCIe without touching Host RAM [cite: 26, 28]
    cudaMemcpyPeer(d_1, dev1, d_0, dev0, size);

    std::cout << "Direct P2P copy complete." << std::endl;

    // Cleanup
    cudaFree(d_0);
    cudaFree(d_1);
    return 0;
}
