
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include "vector_ops.h"




int suma_secuencial(float *h_A, float *h_B, int size, int veces);
int suma_paralela(float *d_A, float *d_B, int size, int veces);

int main(int argc, char *argv[])
{
   
    int n,veces;

    if( argc != 3 ) {
          printf("Usar %s <veces> <tamanio> \n", argv[0]);
          exit(-1);

    }
   

    /* detecto placa y su nombre */
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Computer name: %s \n ", deviceProp.name);
 

    veces = atoi(argv[1]);
    n = atoi(argv[2]);
    
    /* alocacion de memoria en host */
    float *h_A = (float *) malloc(n * sizeof(float));
    float *h_B = (float *) malloc(n * sizeof(float));
    float *h_aux = (float *) malloc(n * sizeof(float));  

    /* alocacion de memoria en device */
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, sizeof(float) * n); 
    cudaMalloc((void**)&d_B, sizeof(float) * n);  

  
    /* chequeo de alocacion de memoria */
    if (!h_A || !h_B || !d_A || !d_B || !h_aux) {
        printf("Error alocando vectores \n");
        exit(-1);
    }

    /* inicializacion de vectores */
    printf("Inicializacion vector A \n");
    vector_inicializacion_random(h_A, n);
    printf("Inicializacion vector B \n");
    vector_inicializacion_random(h_B, n);

 
    /* transferencia de datos cpu -> gpu (host -> device) */
    cudaMemcpy(d_A, h_A, sizeof(float) * n, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, sizeof(float) * n, cudaMemcpyHostToDevice); 

    /* suma secuencial */ 
    printf("Suma secuencial (CPU)\n");
    suma_secuencial(h_A, h_B, n, veces);
  
    /* suma paralela */
    printf("Suma paralela (GPU) \n");
    suma_paralela(d_A, d_B, n, veces);

    /* traigo los datos desde GPU a CPU para testear la suma */
    cudaMemcpy(h_aux, d_A, sizeof(float) * n, cudaMemcpyDeviceToHost);

    /* se chequea el ultimo resultado, despues de sumar VECES veces*/
    if(vector_iguales(h_aux, h_A, n)) 
        printf("Test pasado! \n");
    else
        printf("Test no pasado! \n");

      
    /* liberacion de memoria */
    free(h_A);
    free(h_B);
    free(h_aux);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}



int suma_secuencial(float *h_A, float *h_B, int size, int veces)
{
    
    /* tomar el tiempo inicial */
    struct timeval start;
    gettimeofday(&start, NULL);

    
    int i;
    for(i = 0; i < veces; i++)
    {
        vector_suma_sec(h_A, h_B, size);
    }

    /* tomar el tiempo final */
    struct timeval finish;
    gettimeofday(&finish, NULL);

    /* imprimir el tiempo transcurrido */
    double time = ((finish.tv_sec - start.tv_sec) * 1000.0) + ((finish.tv_usec - start.tv_usec) / 1000.0);
    printf("Tiempo en CPU: %g ms \n", time);


    return 1;
}


int suma_paralela(float *d_A, float *d_B, int size, int veces)
{ 
   
    /* variables para tomar el tiempo en GPU: events */
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* tomar el tiempo inicial */
    cudaEventRecord(start,0);

    int i;
    for(i = 0; i < veces; i++)
    {
        vector_suma_par(d_A, d_B, size);
    }

    /* tomar el tiempo final y calcular tiempo transcurrido */ 
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Tiempo en GPU: %g ms \n", elapsedTime);

    return 1;
}


