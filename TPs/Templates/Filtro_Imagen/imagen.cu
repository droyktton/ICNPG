#include <stdio.h>
#include <stdlib.h>

#include "imagen.h"

/***************************************************/
/* ENTRADA Y SALIDA DE IMAGENES                    */
/***************************************************/

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



/* Esta funcion esta hecha especialmente para ser usada por GNUPLOT que invierte las matrices */
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

			if ((i == 0) || (i == filas-1) || (j == 0) || (j == cols-1)) 
				dato = 0.0;
			else
				dato = imagen[(filas-i)*cols + j];  // arranco desde abajo porque gnuplot grafica al reves
			fprintf(output,"%.0f ",dato); 
			
		}
		fprintf(output, "\n");
	}
	fclose(output);

}


/***************************************************/
/* FILTRO SECUENCIAL                               */
/***************************************************/

void filtro_sec_promedio(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro)
{
	int i,j,k,l;
	float aux;

	/* TODO: Procesar cada pixel de la imagen. Para simplificar, no se procesan los bordes ya que el filtro utiliza los 8 vecinos */
	for (i = 1; i < filas-1; i++) {
		for (j = 1; j < cols-1; j++){
			aux = 0.0;

			/* TODO: aplique el filtro al pixel [i,j] */
			//...
	

	
			// modifico la imagen
			imagen_out[i*cols + j] = //... ;
			
			
		}  // for columnas (j)
	} // for filas (i)

}




/* inicializacion del filtro  */
void inicializar_filtro_promedio(float *filtro, int tamFiltro)
{
	int i;

	for (i=0; i < tamFiltro; i++)
		filtro[i] = 1.0;

}



/********************************************************************************/
/* 		FILTRO 	PARALELO 										                */
/********************************************************************************/


__global__ void kernel_aplicar_filtro_promedio(float * d_imagen_in, float * d_imagen_out, float *d_filtro, int filas, int cols)
{

	/* TODO: obtenga la fila y columna del thread que define el pixel a procesar */
   //int myCol = ...;   // obtiene columna del thread 
   //int myRow = ...;  // obtiene fila del thread
 

   /* TODO: Procesar cada pixel de la imagen. Para simplificar, no se procesan los bordes ya que el filtro utiliza los 8 vecinos */
   if ((myCol < cols-1) && (myRow < filas -1) && (myCol > 0) && (myRow > 0)) {

   	   	int k, l;
   		float aux = 0.0;


		/* TODO: aplique el filtro al pixel [myRow,myCol] */
			//...
		
   		/* TODO: escriba la imagen de salida con el promedio */
		// d_imagen_out[myRow*cols + myCol] = ...; 
	}	
			

}





/* filtro paralelo */
void filtro_par_promedio(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro)
{
	
	/* TODO: lanzamiento del kernel, se crea 1 thread por pixel de la imagen en una grilla 2D que 
	         tiene el mismo tamanio que la imagen a procesar. */
	dim3 nThreads(16,16);
	dim3 nBlocks(cols/nThreads.x + (cols % nThreads.x ? 1 : 0), filas/nThreads.y + (filas % nThreads.y ? 1 : 0));

	/* TODO: complete el lanzamiento del kernel: 1 thread por pixel de la imagen */
	// kernel_aplicar_filtro_promedio<<<..., ...>>>(d_imagen_in, d_imagen_out, d_filtro, filas, cols);
	cudaDeviceSynchronize();

}




