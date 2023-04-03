#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>

#include "imagen.h"
#include "cpu_timer.h"
#include "gpu_timer.h"

#define Y 750
#define X 499
#define TAM_FILTRO 9

#define NAME "Antonov.txt"

#define SALVAR_IMAGEN 1


int main()
{

	int size = X * Y * sizeof(float);

	/* Alocacion de memoria en host */ 
	float *h_imagen_in, *h_imagen_out, *imagen_out_check;
	h_imagen_in = (float*) malloc(size);
	h_imagen_out = (float*) malloc(size);
	imagen_out_check =(float*)malloc(size);

    const char OUTPUT_SEC[] = "output_sec.txt";
	const char OUTPUT_PAR[] = "output_par.txt";

	/* Alocacion de memoria en device */ 
	float *d_imagen_in, *d_imagen_out;
	cudaMalloc((void**)&d_imagen_in, size);
	cudaMalloc((void**)&d_imagen_out, size);
	


	if (!h_imagen_in || !h_imagen_out || !d_imagen_in || !d_imagen_out  ) {
		printf("No aloca memoria para la imagen o filtro \n");
		exit(-1);
	}


	/* LECTURA DE IMAGEN A PROCESAR */
	leer_imagen(NAME, h_imagen_in, Y, X);
	/* copia de datos desde CPU a GPU: imagen original y filtro*/
	cudaMemcpy(d_imagen_in, h_imagen_in, size, cudaMemcpyHostToDevice);



/* FILTRO SOBEL SECUENCIAL */
	cpu_timer crono_cpu;
	crono_cpu.tic();
	aplicar_filtro_sobel_sec(h_imagen_in, imagen_out_check, Y, X);
	crono_cpu.tac();

	if (SALVAR_IMAGEN)
		salvar_imagen(OUTPUT_SEC, imagen_out_check, Y, X);


/* FILTRO SOBEL PARALELO*/
	gpu_timer crono_gpu;
	crono_gpu.tic();
	aplicar_filtro_sobel_par(d_imagen_in, d_imagen_out, Y, X);
	crono_gpu.tac();


	cudaMemcpy(h_imagen_out, d_imagen_out, size, cudaMemcpyDeviceToHost);

	if (SALVAR_IMAGEN)
		salvar_imagen(OUTPUT_PAR, h_imagen_out, Y,X);


	printf("Sobel -> [(Filas x Columnas) imagen /ms_cpu/ms_gpu]= (%dx%d) %lf %lf\n", Y,X, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed);


	free(h_imagen_in);
	free(h_imagen_out);
	free(imagen_out_check);

	cudaFree(d_imagen_in);
	cudaFree(d_imagen_out);


	return 0;
}
