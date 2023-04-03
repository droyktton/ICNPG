#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "gpu_timer.h"
#include "cpu_timer.h"


/* Size of the input data */
#define N 8388608  
/* Size of the filter */
#define M 64


/* Floating point type */
typedef float FLOAT;

/* Function to setup the filter */
void SetupFilter(FLOAT* filter, int size) {
	for(int i = 0 ; i < size ; i++)
		filter[i] = 1.0/size;
}


/* convolucion en la cpu: requiere dos loops */
void conv_sec(FLOAT* input, FLOAT* output, FLOAT * filter) 
{
	FLOAT temp;
	for(int j=0;j<N;j++){
		temp=0.0;
		for(int i=0;i<M;i++){
	  		temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}

// TODO: convolucion en GPU
__global__ void conv_par(FLOAT* input, FLOAT* output, FLOAT* filter) 
{	  	
// su codigo aquí ....
}

////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
    cudaDeviceProp deviceProp;
    int dev; cudaGetDevice(&dev);
    
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);


	/* Allocate memory on host */
	FLOAT *h_input = (FLOAT *) malloc((N+M) * sizeof(FLOAT));  /* Input data */
	FLOAT *h_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* Output data */
	FLOAT *check_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* salida de CPU */
	/* Allocate memory for filter */
	FLOAT *h_filter = (FLOAT*) malloc(M * sizeof(FLOAT));


	/* Setup the filter */
	SetupFilter(h_filter, M);

	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) 
		h_input[i] = (FLOAT)(rand() % 100); 
	
	for(int i = N ; i < N+M ; i++) 
		h_input[i] = h_input[i-N];


	/* Corresponding Device Arrays */
	FLOAT *d_input, *d_output, *d_filter;
	/* TODO: Aloque memoria de device para los arrays */
	//....
	
	/* TODO: Copie el input array al device */
	//....

	/* TODO: Copie el filtro a la GPU */
	//....

	/* check in the CPU */
	cpu_timer crono_cpu; 
	crono_cpu.tic();
	conv_sec(h_input, check_output, h_filter);
	crono_cpu.tac();

	/* TODO: configure el lanzamiento del kernel de convolución	
	// dim3 block_size(...);
  	// dim3 grid_size(...);

	/* lance el kernel de convolución */
	gpu_timer crono_gpu;
	crono_gpu.tic();
	// conv_par<<<grid_size, block_size>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono_gpu.tac();
	
	printf("[M/N/ms_cpu/ms_gpu]= %d %d %lf  %lf  \n", M, N, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed);

	/* TODO: Copiar el output array al host */
	//....
	checkCUDAError("Memcpy output array : ");

	/* comparación */
	FLOAT error, maxerror;
	for(int j=0;j<N;j++){
		error = fabs(h_output[j]-check_output[j]); 
		if(maxerror<error) maxerror=error;
	}
	printf("error maximo= %lf \n \n", maxerror);

	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);
	free(check_output);

	/* TODO: Free memory on device */
	// su código aquí ...	
}

