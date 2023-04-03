#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "gpu_timer.h"
#include "cpu_timer.h"



/* Tamanio del array de input */
const int N = 512*32; 

/* Tamanio del filtro */
const int M = 32*32;

/* Floating point type */
typedef float FLOAT;
//typedef double FLOAT;


/* Prototipo de funciones auxiliares */

/* Inicializacion del filtro */
void SetupFilter(FLOAT* filter, size_t size);
/* Convolucion secuencial  */
void conv_cpu(const FLOAT* input, FLOAT* output, const FLOAT* filter);
/* Convolucion paralela memoria global */
__global__ void conv_gpu (const FLOAT* input, FLOAT* output, const FLOAT* filter);
/* Convolucion paralela memoria de constantes */
__global__ void conv_gpu_constant_memory (const FLOAT* input, FLOAT* output);
/* Convolucion paralela en memoria compartida */
__global__ void conv_gpu_shared_memory(const FLOAT *input, FLOAT *output, const FLOAT *filter);



int main() 
{

	/* Imprime input/output general */
	printf("Input size N: %d\n", N);
	printf("Input size M: %d\n", M);

	/* se imprime el nobre de la placa */
	int card;
	cudaGetDevice(&card);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, card);
	printf("\nDevice %d: \"%s\" \n", card, deviceProp.name);

	/* chequeos de dimensiones de senial y filtro */
	/* version validapara N multiplo de M         */ 
	assert(N % M == 0);
	assert(M <= 1024);


	/* TODO: aloque memoria en host para el input, output, check output y filtro  */
	FLOAT *h_input, *h_output, *check_output, *h_filter;
	// h_input = ...;  	/* Vector Input -> N con artificio Nh para evitar overfloat*/
	// h_output = ...; 			/* Vector Output -> N. Se usa para como salida para las 3 implementaciones en GPU  */
	// check_output = ...; 		/* Check-Output -> N. Se usa como salida de la solucion secuencial */
	// h_filter = ...;			/* Vector filtro ->M */


	/* TODO: chequear correcta alocacion de memoria en CPU */

	/* Inicializa el filtro */
	SetupFilter(h_filter, M);

	/* Llena el array de input (CON "padding") con numeros aleatorios acotados */
	for(int i = 0 ; i < N+M ; i++){
		h_input[i] = (FLOAT)(rand() % 100); 
	}

	/* TODO: alocar memoria en device para el input, filtro, y los outputs de las 3 versiones en GPU */
	FLOAT *d_input, *d_output, *d_filter, *d_output_sm, *d_output_cm;
	//cudaMalloc((void **) &d_input,..));  		// senial
	//cudaMalloc((void **) &d_filter, ... );  	// filtro
	//cudaMalloc((void **) &d_output, ... );   	// salida convolucion paralela normal
	//cudaMalloc((void **) &d_output_sm, ... );  	// salida usando memoria compartida
	//cudaMalloc((void **) &d_output_cm, ... );  	// salida usando memoria de constantes
    

	// TODO: chequear correcta alocacion de memoria en GPU


	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
	cudaMemset(d_output_sm,0,N * sizeof(FLOAT));
	cudaMemset(d_output_cm,0,N * sizeof(FLOAT));

	/* TODO: copiar senial de entrada (h_input) y filtro en GPU*/
	//cudaMemcpy(...);
	//cudaMemcpy(...);
	
	/* cronometraje */
	cpu_timer crono_cpu; 
	crono_cpu.tic();

	/* convolucion en la CPU */
	conv_cpu(h_input, check_output, h_filter);

	crono_cpu.tac();

	/****************************************************/
	/* VERSION PARALELA  -  FILTRO EN MEMORIA GLOBAL    */
	/****************************************************/

  	/*Defino tamaño bloque y grilla */
  	dim3 block_size(M);
  	dim3 grid_size(N/M);

  	gpu_timer crono_gpu;
  	crono_gpu.tic();
    
    /* TODO: lanzamiento del kernel que usa memoria global para filtro y senial. Salida queda en d_output */
	// conv_gpu<<< ..., ...>>>(d_input, d_output, d_filter);

	crono_gpu.tac();
	
	/* TODO: copiar el resultado de device a host usando h_output y d_output */
	//cudaMemcpy( ... );

	/* Comparacion (lea documentacion de la funcion de C assert si no la conoce)*/	
	for(int j=0; j<N; j++){
		assert(h_output[j] == check_output[j]);
	}



	/*****************************************************/
	/* VERSION PARALELA  -  FILTRO EN MEMORIA COMPARTIDA */
	/*****************************************************/

	gpu_timer crono_gpu_sm;
  	crono_gpu_sm.tic();
   
   /* TODO: lanzamiento del kernel que usa memoria compartida para filtro. Salida queda en d_output_sm */
	// conv_gpu_shared_memory<<< ... , ... >>>(d_input, d_output_sm, d_filter);

	crono_gpu_sm.tac();

	/* TODO: copiar el resultado de device a host usando h_output y d_output_sm */
	//cudaMemcpy( ... );
	

	/* Comparacion (lea documentacion de la funcion de C assert si no la conoce)*/
	for(int j=0; j<N; j++){
		assert(h_output[j] == check_output[j]);
	}


	/****************************************************/
	/* VERSION PARALELA  -  MEMORIA DE CONSTANTES 		*/
	/****************************************************/

	/* TODO: copiar h_filter en d_filtro constante */
	// cudaMemcpyToSymbol(...);

	gpu_timer crono_gpu_cm;
  	crono_gpu_cm.tic();
   
   /* TODO: lanzamiento del kernel que usa memoria de constantes para filtro. Salida queda en d_output_cm */
	//conv_gpu_constant_memory<<<...,...>>>(d_input, d_output_cm);

	crono_gpu_cm.tac();


	/* TODO: copiar el resultado de device a host usando h_output y d_output_cm */
	//cudaMemcpy(...);


	/* Comparacion (lea documentacion de la funcion de C assert si no la conoce)*/
	for(int j=0; j<N; j++){
		assert(h_output[j] == check_output[j]);
	}

	
	
/* Impresion de tiempos */
	printf("[N/M/ms_cpu/ms_gpu/ms_gpu_sm/ms_gpu_cm]= [%d/%d/%lf/%lf/%lf/%lf] \n", N, M, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed, crono_gpu_sm.ms_elapsed, crono_gpu_cm.ms_elapsed);


   

	/* TODO: liberer memoria en host y device */
	//...
	
	return(0);
}



/* Inicializacion del filtro */
void SetupFilter(FLOAT* filter, size_t size) 
{
	/* TODO: llene el filtro con el valor que desee*/

	for(int i = 0; i < size; i++){
		// filter[i] = ...; 
	}
}


/* Convolucion secuencial  */
void conv_cpu(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{
	/* TODO: implemente la versión secuencial de la convolucion */
	FLOAT temp;
	/*Barro vector input (tamaño N) y para cada elemento j hasta N hago la operacion*/
	/*de convolucion: elemento i del vector filter por elemento i+j del vector input */
	//...

}


// declaracino del filtro en memoria constante
__constant__ FLOAT d_filtro_constant[M];

__global__ void conv_gpu_constant_memory (const FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  /* TODO: resuelva la convolucion utilizando el filtro en memoria constante */

	  // ...
}


// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output todo en memoria global
__global__ void conv_gpu (const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	/*Barro vector input (tamaño N) y para cada elemento j hasta N hago la operacion*/
	/*de convolucion: elemento i del vector filter por elemento i+j del vector input */

	/*TODO: descomente y complete para resolver la convolucion paralela*/
//	output[j]= ...;
//	for(int i=0; i<M; i++){
	  //...;
//	}	
}



/* Solucion que solo sirve para Nh menor a tamaño de bloque */
__global__ void conv_gpu_shared_memory(const FLOAT *input, FLOAT *output, const FLOAT *filter)
{

	int tidx = blockIdx.x * blockDim.x + threadIdx.x; // global
	int id = threadIdx.x;

	__shared__ float filter_sm[M];
	
	/* TODO: lleno el vector de memoria compartida con los datos del filtro*/
	if (id < M)
	//	filter_sm[id] = ;

	// todos los threads del bloque se deben sincronizar antes de seguir	
	syncthreads();


	/*Barro vector input (tamaño N) y para cada elemento j hasta N hago la operacion*/
	/*de convolucion: elemento i del vector filter por elemento i+j del vector input*/
	/* TODO: realicela convolucion usando el filtro en memoria compartida */	
	output[tidx]=0.0;
	for(int i=0; i<M; i++){
	  	// output[tidx] += ...; 
	}	

}
