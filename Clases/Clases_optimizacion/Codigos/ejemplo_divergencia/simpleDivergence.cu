#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include "gpu_timer.h"

/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 *
 *  warpSize: constante predefinida de CUDA. Valor 32 hasta que esto cambie...
 */

#define VECES 10


// THREAD approach
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    // THREADS pares ejecutan then, THREADS impares else. 
    // Divergencia!
    if (tid % 2 == 0)  
    {
	for(int i=0;i<100;i++)
        ia += sqrt(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }
    else
    {
	for(int i=0;i<100;i++)
        ib += 2.0*sqrt(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }
    c[tid] = ia + ib;
}


// WARP approach
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    // WARPS pares ejecutan el then, WARPS impares el else
    if ((tid / warpSize) % 2 == 0)  
    {
	for(int i=0;i<100;i++)
        ia += sqrt(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }
    else
    {
	for(int i=0;i<100;i++)
        ib += 2.0*sqrtf(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }

    c[tid] = ia + ib;
}



__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)  
    {
	for(int i=0;i<100;i++)
        ia += sqrt(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }
    else
    {
	for(int i=0;i<100;i++)
        ib += 2.0*sqrtf(sinf(i)*sinf(i)+cosf(i)*cosf(i));
    }

    c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\n%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blocksize = 64;

    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size      = atoi(argv[2]);

    printf("Data size %d ", size);
    assert(size%blocksize==0);

    // set up execution configuration
    dim3 block (blocksize, 1,1);
    dim3 grid  ((size + block.x - 1) / block.x, 1,1);
    printf("Execution Configure (block %d grid %d) \n\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // run a warmup kernel to remove overhead
    cudaDeviceSynchronize();
    gpu_timer crono_gpu;
    crono_gpu.tic();

    warmingup<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    crono_gpu.tac();

   
    cudaGetLastError();

    // run kernel 1
    crono_gpu.tic();
    
    int i;
    for (i = 0; i < VECES; i++) {
        mathKernel1<<<grid, block>>>(d_C);
        
    }
    cudaDeviceSynchronize();
    crono_gpu.tac();

    printf("Thread approach: mathKernel1 <<< %d %d >>> elapsed %lf msecs \n\n", grid.x, block.x, crono_gpu.ms_elapsed);
    cudaGetLastError();

    // run kernel 3
    crono_gpu.tic();
    for (i = 0; i < VECES; i++) {
        mathKernel2<<<grid, block>>>(d_C);   
    }
   
    cudaDeviceSynchronize();
    crono_gpu.tac();

    printf("Warp approach: mathKernel2 <<< %d %d >>> elapsed %lf msecs\n\n", grid.x, block.x, crono_gpu.ms_elapsed);
    cudaGetLastError();
   

    // free gpu memory and reset divece
    cudaFree(d_C);
    return (0);
}


/*    COMPILACION  */
/*  nvcc -O3 -arch=sm_21 simpleDivergence.cu -o simpleOpt
    nvcc -g -G -arch=sm_21 simpleDivergence.cu -o simple

    usar ./simpleOpt y ./simple para ver tiempos similares

    usar 

    nvprof --metrics branch_efficiency ./simple
    nvprof --metrics branch_efficiency ./simpleOpt

    */
