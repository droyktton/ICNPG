#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//util para el profiling!
#include <nvToolsExt.h>


#define NUM_BINS 10
#define NUM_SAMPLES 1000000

__global__ void compute_histogram(float *d_samples, int *d_histogram, int num_samples, float bin_width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_samples) {

        int bin_index = (int)(d_samples[tid] / bin_width);

        //esto seria incorrecto, por la condición de carrera
        //d_histogram[bin_index]++;

        //evitamos asi la condicion de carrera
        atomicAdd(&d_histogram[bin_index], 1);
    }
}

int main() {
    float *h_samples, *d_samples;
    int *h_histogram, *d_histogram;

    // Allocate memory for samples and histogram on host and device
    h_samples = (float *)malloc(sizeof(float) * NUM_SAMPLES);
    h_histogram = (int *)malloc(sizeof(int) * NUM_BINS);
    cudaMalloc((void **)&d_samples, sizeof(float) * NUM_SAMPLES);
    cudaMalloc((void **)&d_histogram, sizeof(int) * NUM_BINS);

    // Initialize samples on host
    for (int i = 0; i < NUM_SAMPLES; i++) {
        h_samples[i] = (float)rand() / RAND_MAX;
    }

    // Copy samples from host to device
    cudaMemcpy(d_samples, h_samples, sizeof(float) * NUM_SAMPLES, cudaMemcpyHostToDevice);

    // Compute bin width
    float bin_width = 1.0f / NUM_BINS;

    nvtxRangePush("-----Histograma en Device-----");

    // Launch kernel to compute histogram
    compute_histogram<<<(NUM_SAMPLES + 255) / 256, 256>>>(d_samples, d_histogram, NUM_SAMPLES, bin_width);
    cudaDeviceSynchronize();

    nvtxRangePop();


    // Copy histogram from device to host
    cudaMemcpy(h_histogram, d_histogram, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);

    // Print histogram
    for (int i = 0; i < NUM_BINS; i++) {
        printf("%d %d\n", i, h_histogram[i]);
    }

    // Free memory on host and device
    free(h_samples);
    free(h_histogram);
    cudaFree(d_samples);
    cudaFree(d_histogram);

    return 0;
}
