/*
Demuestra el uso de directivas de openacc
para compilar:

pgc++ -acc -Minfo convolucion_openacc.c

el timming ahora incluye copias H->D->H
*/

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "simple_timer.h"


/* Size of the input data */
//#define N 8388608  
#define N 33554432
/* Size of the filter */
#define M 1024


/* Floating point type */
typedef float FLOAT;

/* Function to setup the filter */
void SetupFilter(FLOAT* filter, int size) {
	for(int i = 0 ; i < size ; i++)
		filter[i] = 1.0/size;
}


/* convolucion en cpu o gpu segun compilacion */
void conv(const FLOAT* __restrict__ input, FLOAT* __restrict__ output, const FLOAT * __restrict__ filter) 
{
	FLOAT temp;
	#pragma acc kernels copyin(input[0:N],filter[0:M]) copyout(output[0:N]) 
	//#pragma acc parallel loop copyin(input[0:N],filter[0:M]) copyout(output[0:N]) 
	#pragma omp parallel for private(temp) shared(filter,input,output) 
	for(int j=0;j<N;j++){
		temp=0.0;
		for(int i=0;i<M;i++){
	  		temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}


////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
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

	/* convolution */
	omp_timer crono_cpu; 
	crono_cpu.tic();
	conv(h_input, check_output, h_filter);
	crono_cpu.tac();


	printf("[M/N/ms_cpu/ms_gpu]= %d %d %lf\n", M, N, crono_cpu.ms_elapsed);

	//for(int i=0;i<10;i++) 
	//	printf("%lf\n", check_output[i]);

	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);
	free(check_output);	
}

