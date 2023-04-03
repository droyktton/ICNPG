#include <stdio.h>
#include <stdlib.h>

#include "imagen.h"


/* ENTRADA Y SALIDA DE MAPAS */
void leer_imagen(const char * nombre, float *imagen, int filas, int cols)
{
	FILE *input = fopen(nombre,"r");

	if (input == NULL)
	{
		printf("Error en leer_imagen: no abre archivo %s \n", nombre);
		exit(-1);
	}

	int i,j, dato;


	for(i = 0; i < filas; i++)
	{
		for(j = 0; j < cols; j++)
		{
			fscanf(input, "%d", &dato);
			imagen[i*cols + j] = (float)dato;
		}
		
	}
	fclose(input);

}


void salvar_imagen(const char * nombre, float *imagen, int filas, int cols)
{
	FILE *output = fopen(nombre,"w");

	if (output == NULL)
	{
		printf("Error en salvar: no abre archivo %s \n", nombre);
		exit(-1);
	}

	printf("Escribiendo en archivo: %s \n", nombre);

	int i,j;
	float dato;
	for(i = 0; i < filas; i++) {
		for(j = 0; j < cols; j++) {
			dato = imagen[(filas-i)*cols + j];  // arranco desde abajo por gnuplot
			fprintf(output,"%.0f ",dato); 
			
		}
		fprintf(output, "\n");
	}
	fclose(output);

}



/* filtro secuen*/
void filtro_sec(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro)
{
	int i,j,k,l;
	float aux;

	for (i = 1; i < filas-1; i++)
		for (j = 1; j < cols-1; j++){
			aux = 0.0;

			// aplico el filtro 8 vecinos	
			for(k=-1; k <= 1; k++) { // fila filtro
				for (l = -1; l <= 1; l++) {// col filtro
					//aux = aux + imagen_in[...] * filtro[...];
					
				}
			}
			// modifico la imagen
			//imagen_out[...] = (float)... ; // casting a float	
		}
}




/* Inicializar el filtro horizontal usado para calcular Gx */
void inicializar_filtro_sobel_horizontal(float *filtro)
{

//	filtro[0] = -1.0;
//	filtro[0] = -1.0;
//	filtro[1] = ...
//  ...
//	filtro[8] = ...// 
}


/* Inicializar el filtro vertical usado para calcular Gy */
void inicializar_filtro_sobel_vertical(float *filtro)
{
//	filtro[0] = -1.0;
//	filtro[1] = ...
//  ...
//	filtro[8] = ...
}


void aplicar_filtro_sobel_sec(float *imagen_in, float *imagen_out, int filas, int cols)
{
	float *filtro_hor, *filtro_ver, *g_x, *g_y;

	int dim = filas * cols; 

	/* Alocacion de memoria para filtros y matrices g  */ 
	//filtro_hor = ... 
	//filtro_ver = ... 
	//g_x = ...
	//g_y = ...

	/* Chequear  que la memoria haya sido alocada */
	if (!filtro_hor || !filtro_ver || !g_x || !g_y) {
		printf("No aloca arreglos \n ");
		exit (-1);
	}


	/* Inicializacion de filtros. Son distintos por eso distintas funciones */
	inicializar_filtro_sobel_vertical(filtro_ver);
	inicializar_filtro_sobel_horizontal(filtro_hor);

	/* Aplicacion de filtros para obtener g */
	filtro_sec(imagen_in, g_x, filas, cols, filtro_hor);
	filtro_sec(imagen_in, g_y, filas, cols, filtro_ver);

	/* calculo final para obtener los bordes */ 
	calcular_g(g_x, g_y, imagen_out, filas, cols);

	/* Liberacion de memoria */
	//free(...

}


/* Combinacion gradientes horizontal y vertical para obtener los bordes*/
void calcular_g(float *g_x, float *g_y, float *imagen_out, int filas, int cols)
{
    int i,j;

    for (i = 0; i < filas; i++)
    	for (j = 0; j < cols; j++)
	//		imagen_out[i*cols + j] = (float)sqrt(...


}



/********************************************************************************/
/* 								PARALELO 										*/
/********************************************************************************/


__global__ void kernel_aplicar_filtro(float * d_imagen_in, float * d_imagen_out, float *d_filtro, int filas, int cols)
{

   int myCol = threadIdx.x + (blockDim.x * blockIdx.x);
   int myRow = threadIdx.y + (blockDim.y * blockIdx.y);

   // no aplico filtro a los bordes
   if ((myCol < cols-1) && (myRow < filas -1) && (myCol > 0) && (myRow > 0)) {

   	   	int k, l;
   		float aux = 0.0;


		// aplico el filtro 8 vecinos	
		for(k=-1; k <= 1; k++) { // fila filtro
			for (l = -1; l <= 1; l++) {// col filtro
				//...
			}
		}
		// modifico la imagen
		//d_imagen_out[... ;
	}	
			

}


__global__ void kernel_calcular_g(float *d_g_x, float *d_g_y, float *d_imagen_out, int filas, int cols) 
{

   int myCol = threadIdx.x + (blockDim.x * blockIdx.x);
   int myRow = threadIdx.y + (blockDim.y * blockIdx.y);
   int myId = myRow * cols + myCol;

   if ((myCol < cols-1) && (myRow < filas -1) && (myCol > 0) && (myRow > 0)) {
   	//...		d_imagen_out[myId] = ...

}


/* filtro paralelo */
void filtro_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro)
{
	// Preparar variables para definir la grilla del kernel
	//dim3 ...
	//dim3 ...

	//kernel_aplicar_filtro<<<... 
	cudaDeviceSynchronize();

}



/*  SOBEL paralelo*/
void aplicar_filtro_sobel_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols)
{
	// Aplicar el filtro de forma paralela. Guiarse del código secuencial. 


}