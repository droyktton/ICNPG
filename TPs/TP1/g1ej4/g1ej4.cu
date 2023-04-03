/* 
   Guia 1, Ej 4
   ICNPG 2016
 
   Complete el template para obtener la integral mediante el metodo de Simpson.
*/

#include <stdlib.h>
#include <stdio.h>

#include "../../common/curso.h"


// Number of kernel runs
#define NITER 100   
/*
 *  Esta función calcula un vector con integrales parciales,
 *  una por cada elemento de x,
 *  que se reducen a la integral total en CPU.
 */

extern "C"{
__global__ void simpson(float* x, float *data, float *integral, int N)
 {
    int index = threadIdx.x + blockIdx.x * blockDim.x ;
    float tmp;
    float h;
    float xi,xim,yi,yim;

    // compute the average of this thread's left and right neighbors
 
    // 
    // Inserte su código aqui
    // 
    
 }
}

/*
 *  Esta función calcula un vector con integrales parciales,
 *  una por cada bloque de threads,
 *  que se reduce a la integral total en CPU.
 */
extern "C"{
__global__ void simpson_gpu(float* x, float *data, float *integral, int N)
 {
    int index = threadIdx.x + blockIdx.x * blockDim.x ;
    int tid = threadIdx.x;
    float tmp;
    float h;
    float xi,xim,yi,yim;
    
    /*
    First step: Compute partial sums
        Compute the average of this thread's left and right neighbors
    */
 
    // 
    // Inserte su código aqui
    // 
    
    /*
    Second step: Reduction of sum_block
    */    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, numthreads must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (tid < i)
            integral[index] += integral[index + i];
        __syncthreads();
        i /= 2;
    }
    
 }
}

int main( int argc, const char** argv ) {
    float *x, *y, *integ;
    float *d_x, *d_y, *d_integ;
    int N,numthreads;

    if(argc==1){ 
        printf("Usage %s [array size] [number of threads per block]\n",argv[0]);
        exit(1);
    }
    if(argc==2){ 
        printf("Assuming number of threads per block=32\n");
        N = atoi(argv[1]);
        numthreads = 32;
    }
    if(argc==3){ 
        N = atoi(argv[1]);
        numthreads = atoi(argv[2]);
    }
    printf(" Threads per Block: %d \n", numthreads);
    // Check if the number of blocks is compatible with a reduction
    // use a bit operation trick to test if numblock is a power of two.
    int numblocks = (N+numthreads-1)/numthreads;
    if((numblocks & (numblocks - 1)) != 0) {   
        printf("Es necesario que numblocks (%d) sea potencia de dos.\n",numblocks);
        return 1;
    }        
    
    printf(" Blocks  per  Grid: %d \n", numblocks);
    printf(" Threads per Block: %d \n", numthreads);
    
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    integ = (float*)malloc(N * sizeof(float));

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&d_x, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_y, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_integ, N * sizeof(float) ) );

    float a=0.0;
    float b=3.0;
    float step = (b-a)/(N-1);

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        x[i] = a+i*step;
        y[i] = sin(x[i]);
    }

    // copy the arrays 'x' and 'y' to the GPU
    HANDLE_ERROR( cudaMemcpy( d_x, x, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_y, y, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );


    dim3 dimGrid(numblocks);
    dim3 dimBlock(numthreads);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    for (int i=0; i<NITER; i++) 
        simpson<<<dimGrid,dimBlock>>>( d_x, d_y, d_integ ,N);
    checkCUDAError("Launch failure: ");

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop ); //Necesario

    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,start, stop ); // en milisec.

    for(int i=0;i<N;i++) 
        integ[i] = 0.0;

    // copy the array of partial sums back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( integ, d_integ, N * sizeof(float),
                              cudaMemcpyDeviceToHost ) );

    // compute the integral from partial sums
    float sum=0;
    // Last reduction from data of blocks
    // 
    // Inserte su código aqui
    // 

    // display the results
    printf( "\nExact: %f  Simpson: %f\n", 1-cos(x[N-1]), sum );
    printf( "\nSize: %d  Blocks: %d  Threads per Block: %d  numthreads: %d  Time in Kernel:  %3.1f ms\n",
     N, numblocks, numthreads, numthreads, elapsedTime);

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( d_x ) );
    HANDLE_ERROR( cudaFree( d_y ) );
    HANDLE_ERROR( cudaFree( d_integ ) );

    return 0;
}

