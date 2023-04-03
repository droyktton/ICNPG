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

#define INPUT "Antonov.txt"


#define SALVAR_IMAGEN 1


int main()
{

	
	int size = X * Y * sizeof(float);

	/* Nombres de archivos que se usan si SALVAR_IMAGEN = 1. Se usan luego para ver los resultados */
    const char OUTPUT_SEC[] = "output_sec.txt";
	const char OUTPUT_PAR[] = "output_par.txt";


	/* TODO: alocacion de memoria en host */ 
	float *h_imagen_in, *h_imagen_out, *h_filtro, *imagen_out_check;
	//h_imagen_in = ...;
	//h_imagen_out = ...;
	//h_filtro = ...;
	//imagen_out_check = ...; // se usa para comprar resultados


	/* TODO: Alocacion de memoria en device */ 
	float *d_imagen_in, *d_imagen_out, *d_filtro;
	//...
	//...
	//...


	if (!h_imagen_in || !h_imagen_out || !h_filtro || !d_imagen_in || !d_imagen_out || !d_filtro ) {
		printf("No aloca memoria para la imagen o filtro \n");
		exit(-1);
	}


	/* lectura de la imagen a procesar  */
	leer_imagen(INPUT, h_imagen_in, Y, X);


	/* Inicializacion del filtro en host. Todos 1s*/
	inicializar_filtro_promedio(h_filtro, TAM_FILTRO);


	/* TODO: copiar imagen y filtro desde host a device */
	//... cudaMemcpy(...);
	//... cudaMemcpy(...);
	

	/* Solucion secuencial */
	cpu_timer crono_cpu; 
	crono_cpu.tic();
	
	filtro_sec_promedio(h_imagen_in, imagen_out_check, Y, X, h_filtro);

	crono_cpu.tac();

	/* si SALVAR_IMAGEN es 1 se guarda en disco la imagen */
	if (SALVAR_IMAGEN) 
		salvar_imagen(OUTPUT_SEC, imagen_out_check, Y,X);

	
	/*  Solucion paralela */
	gpu_timer crono_gpu;
	crono_gpu.tic();

	filtro_par_promedio(d_imagen_in, d_imagen_out, Y, X, d_filtro);

	crono_gpu.tac();

	/* TODO: traer los datos desde device a host usando h_imagen_out */
	// cudaMemcpy(h_imagen_out...);


	/* si SALVAR_IMAGEN es 1 se guarda en disco la imagen */
	if (SALVAR_IMAGEN)
		salvar_imagen(OUTPUT_PAR, h_imagen_out, Y , X);

	printf("Filtro -> [(Filas x Columnas) imagen /ms_cpu/ms_gpu]= (%dx%d) %lf %lf\n", Y,X, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed);

	/* Comparacion (lea documentacion de la funcion de C assert si no la conoce)*/	
	/* no proceso los bordes ya que no les aplico el filtro */
	for (int i = 1; i < (Y-1); i++)
		for (int j = 1; j < (X-1); j++){
			assert(h_imagen_out[i * X + j] == imagen_out_check[i * X + j]);
		}
	

	/* TODO: desalocar memoria en host y device de todos los arreglos*/
	//...
	//...

		
	return 0;
}
