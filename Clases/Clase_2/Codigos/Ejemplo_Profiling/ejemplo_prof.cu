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


/* KERNEL 1*/
__global__ void conv_par(FLOAT* input, FLOAT* output, FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<M;i++){
	  		temp += filter[i]*input[i+j];
		}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}



/* kERNEL 2*/
__global__ void suma_par(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  if (j < N) {

	  	FLOAT temp;
	
	  	temp = input[j] + input[j];
	  	output[j] = temp;	
	  }
}



/* KERNEL 3 */
__global__ void operaciones_par(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  
	  if (j < N) {
	  	FLOAT temp;
	
	  	temp = input[j] * input[j];
	  	temp = temp / input[j];
	  	temp = pow(temp,2);
	  	temp = sinf(temp);
	  	output[j] = temp;	
	  }
}


/* KERNEL 4 */
__global__ void operaciones_version_2_par(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  if (j < N) {
	  
	  	FLOAT temp;
	
	 	temp = input[j];
	  	temp = temp / 100;
	 
	 	if (j % 2)
	  		temp = sqrt(temp);
	  	else
	  		temp = cos(temp);
	  	temp = sin(temp);
	  	temp = pow(temp,2);
	  	temp = cos(temp);
	  	output[j] = temp;	
	  }
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
	/* Allocate memory for filter */
	FLOAT *h_filter = (FLOAT*) malloc(M * sizeof(FLOAT));


	/* Setup the filter */
	SetupFilter(h_filter, M);

	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) 
		h_input[i] = (FLOAT)(rand() % 100); 
	
	for(int i = N ; i < N+M ; i++) 
		h_input[i] = h_input[i-N];


	/* Allocate memory on device */
	FLOAT *d_input, *d_output, *d_filter;
	cudaMalloc((void**)&d_input, (N+M) * sizeof(FLOAT));
	cudaMalloc((void**)&d_output, N * sizeof(FLOAT));
	cudaMalloc((void**)&d_filter, M * sizeof(FLOAT));
	
	/* Copy input array to device */
	cudaMemcpy(d_input, h_input, (N+M) * sizeof(FLOAT), cudaMemcpyHostToDevice);

	/* Copy the filter to the GPU */
	cudaMemcpy(d_filter, h_filter, M * sizeof(FLOAT), cudaMemcpyHostToDevice);

	
	//conv_sec(h_input, check_output, h_filter);

	
	dim3 block_size(512);
  	dim3 grid_size(N/block_size.x + (N % block_size.x ? 1 : 0));

	
	conv_par<<<grid_size, block_size>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();


	suma_par<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();
	

	operaciones_par<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();

	operaciones_version_2_par<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();

	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);

	/* Free memory on device */
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);
	
}

