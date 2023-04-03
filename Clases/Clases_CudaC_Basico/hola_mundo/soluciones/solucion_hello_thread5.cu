#include "common.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU(int i)
{
    if(threadIdx.x==i) printf("Hello World from GPU thread %d\n",i);
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    int i=atoi(argv[1]);

    helloFromGPU<<<1, 10>>>(i);
    CHECK(cudaDeviceReset());
    return 0;
}


