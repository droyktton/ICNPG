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
					aux = aux + imagen_in[(i+k)*cols + j+l] * filtro[((k+1)*3) + (l+1)];
					
				}
			}
			// modifico la imagen
			imagen_out[i*cols + j] = (float)aux;// / (float)9 ;
			
			
		}

}



/* inicializacion del filtro  */
/*{
	int i;

	for (i=0; i < tamFiltro; i++)
		filtro[i] = 1.0;

}
*/




void inicializar_filtro_sobel_horizontal(float *filtro, int tamFiltro)
{

	filtro[0] = -1.0;
	filtro[1] = 0.0;
	filtro[2] = 1.0;
	filtro[3] = -2.0;
	filtro[4] = 0.0;
	filtro[5] = 2.0;
	filtro[6] = -1.0;
	filtro[7] = 0.0;
	filtro[8] = 1.0;


}



void inicializar_filtro_sobel_vertical(float *filtro, int tamFiltro)
{
	filtro[0] = -1.0;
	filtro[1] = -2.0;
	filtro[2] = -1.0;
	filtro[3] = 0.0;
	filtro[4] = 0.0;
	filtro[5] = 0.0;
	filtro[6] = 1.0;
	filtro[7] = 2.0;
	filtro[8] = 1.0;


}


void aplicar_filtro_sobel_sec(float *imagen_in, float *imagen_out, int filas, int cols)
{
	float *filtro_hor, *filtro_ver, *g_x, *g_y;

	int dim = filas * cols * sizeof(float);

	filtro_hor = (float*)malloc(9 * sizeof(float));
	filtro_ver = (float*)malloc(9 * sizeof(float));
	g_x = (float*)malloc(dim * sizeof(float));
	g_y = (float*)malloc(dim * sizeof(float));


	if (!filtro_hor || !filtro_ver || !g_x || !g_y) {
		printf("No aloca arreglos \n ");
		exit (-1);
	}


	inicializar_filtro_sobel_vertical(filtro_ver, 9);
	inicializar_filtro_sobel_horizontal(filtro_hor, 9);

	filtro_sec(imagen_in, g_x, filas, cols, filtro_hor);


	filtro_sec(imagen_in, g_y, filas, cols, filtro_ver);

	
	calcular_g(g_x, g_y, imagen_out, filas, cols);

	//salvar_imagen("gx.dat", g_x, filas, cols);

	free(filtro_hor);
	free(filtro_ver);
	free(g_x);
	free(g_y);

}


void calcular_g(float *g_x, float *g_y, float *imagen_out, int filas, int cols)
{
    int i,j;

    for (i = 0; i < filas; i++)
    	for (j = 0; j < cols; j++)
			imagen_out[i*cols + j] = (float)sqrt((float)powf(g_x[i*cols+j],2) + (float)powf(g_y[i*cols+j], 2));    		



}


/********************************************************************************/
/* 								PARALELO 										*/
/********************************************************************************/


__global__ void kernel_aplicar_filtro(float * d_imagen_in, float * d_imagen_out, float *d_filtro, int filas, int cols)
{

   int myCol = threadIdx.x + (blockDim.x * blockIdx.x);
   int myRow = threadIdx.y + (blockDim.y * blockIdx.y);
 //  int myId = myRow * cols + myCol;

   if ((myCol < cols-1) && (myRow < filas -1) && (myCol > 0) && (myRow > 0)) {

   	   	int k, l;
   		float aux = 0.0;


		// aplico el filtro 8 vecinos	
		for(k=-1; k <= 1; k++) { // fila filtro
			for (l = -1; l <= 1; l++) {// col filtro
				aux = aux + d_imagen_in[(myRow+k)*cols + myCol+l] * d_filtro[((k+1)*3) + (l+1)];
				
			}
		}
			// modifico la imagen
		d_imagen_out[myRow*cols + myCol] = (float) aux;// / (float)9 ;
	}	
			

}


__global__ void kernel_calcular_g(float *d_g_x, float *d_g_y, float *d_imagen_out, int filas, int cols) 
{

   int myCol = threadIdx.x + (blockDim.x * blockIdx.x);
   int myRow = threadIdx.y + (blockDim.y * blockIdx.y);
   int myId = myRow * cols + myCol;

   if ((myCol < cols-1) && (myRow < filas -1) && (myCol > 0) && (myRow > 0)) {
   			d_imagen_out[myId] = (float)sqrt((float)powf(d_g_x[myId],2) + (float)powf(d_g_y[myId], 2));    		
   }	

}


/* filtro paralelo */
void filtro_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro)
{
	
	dim3 nThreads(16,16);
	dim3 nBlocks(cols/nThreads.x + 1, filas/nThreads.y + 1);

	kernel_aplicar_filtro<<<nBlocks, nThreads>>>(d_imagen_in, d_imagen_out, d_filtro, filas, cols);
	cudaDeviceSynchronize();

}



/*  SOBEL paralelo*/
void aplicar_filtro_sobel_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols)
{

	int dim = filas * cols * sizeof(float);
	int dimFiltro = 9 * sizeof(float);

	/* inicializacion de filtros, todo en CPU */
	float *h_filtro_hor, *h_filtro_ver, *d_filtro_hor, *d_filtro_ver;
	h_filtro_hor = (float*)malloc(dimFiltro);
	h_filtro_ver = (float*)malloc(dimFiltro);
	cudaMalloc((void**)&d_filtro_hor, dimFiltro);
	cudaMalloc((void**)&d_filtro_ver, dimFiltro);


	/* estos dos resultados intermedios los dejo en device, viven y mueren alli */
	float *d_g_x, *d_g_y;
	cudaMalloc((void**)&d_g_x, dim);
	cudaMalloc((void**)&d_g_y, dim);
	
	if (!h_filtro_hor || !h_filtro_ver || !d_g_x || !d_g_y || !d_filtro_ver || !d_filtro_hor) {
		printf("No aloca arreglos \n ");
		exit (-1);
	}


	inicializar_filtro_sobel_vertical(h_filtro_ver, 9);
	inicializar_filtro_sobel_horizontal(h_filtro_hor, 9);

	cudaMemcpy(d_filtro_hor, h_filtro_hor, dimFiltro, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filtro_ver, h_filtro_ver, dimFiltro, cudaMemcpyHostToDevice);


	dim3 nThreads(16,16);
	dim3 nBlocks(cols/nThreads.x + 1, filas/nThreads.y + 1);

	kernel_aplicar_filtro<<<nBlocks, nThreads>>>(d_imagen_in, d_g_x, d_filtro_hor, filas, cols);
	cudaDeviceSynchronize();

	kernel_aplicar_filtro<<<nBlocks, nThreads>>>(d_imagen_in, d_g_y, d_filtro_ver, filas, cols);
	cudaDeviceSynchronize();

	kernel_calcular_g<<<nBlocks, nThreads>>>(d_g_x, d_g_y, d_imagen_out, filas, cols);
	cudaDeviceSynchronize();


// SECUENCIAL
	//filtro_sec(imagen_in, g_x, filas, cols, filtro_hor);

	//filtro_sec(imagen_in, g_y, filas, cols, filtro_ver);
	
	//calcular_g(g_x, g_y, imagen_out, filas, cols);

	//salvar_imagen("gx.dat", g_x, filas, cols);

	free(h_filtro_hor);
	free(h_filtro_ver);
	
	cudaFree(h_filtro_ver);
	cudaFree(h_filtro_hor);
	cudaFree(d_g_x);
	cudaFree(d_g_y);


}