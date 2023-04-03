
#include "vector_ops.h"
#include <stdio.h>


/*************************************************************/
/*         KERNELS                                           */
/*************************************************************/

/* suma de cada elemento del vector */
__global__ void kernel_suma(float *v1, float *v2, int dim)
{
    int id = threadIdx.x + (blockIdx.x * blockDim.x); 

    if (id < dim)
    {
         v1[id] = v1[id] + v2[id];
    }
}



/*************************************************************/
/*         FUNCIONES C                                       */
/*************************************************************/

/* Suma de vectores (inplace)  */
int vector_suma_sec(float *v1, float *v2, int dim)
{
    for (int i = 0; i < dim; i++) {
        v1[i] = v1[i] + v2[i];
    }

    return 1;
}


/* Suma de vectores. Resultado queda en el primer argumento */
int vector_suma_par(float *v1, float *v2, int dim)
{
    dim3 nThreads(256); 
    dim3 nBlocks((dim / nThreads.x) + (dim % nThreads.x ? 1 : 0));

    kernel_suma<<<nBlocks, nThreads>>>(v1, v2, dim);   

    cudaDeviceSynchronize();    // las invocaciones a kernels son asincronicas (no detienen al thread de CPU que realiza la llamada)

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("kernel error \n");
        exit(-1);
    }

    return 1;
}



/* retorna 1 si los vectores son iguales, 0 cc */
int vector_iguales(float *v1, float *v2, int dim)
{
    int i;
    for(i=0; i < dim; i++) {
        if(v1[i] != v2[i]) 
           return 0;
        
    }

    return 1;
}




/* inicializa el vector pasado como parametro con valores entre 0 y 99 */
int vector_inicializacion_random(float *v, int dim)
{
    srand(time(NULL));

    for (int i = 0; i < dim; i++) {
        v[i] = (float) (rand() % 100);
    }

    return 1;
}


/* imprime el vector pasado como parametro. */
int vector_imprimir(float *v, int dim)
{
    for (int i = 0; i < dim; i++) {
        printf(" %.0f ", v[i]);
    }
    printf(" \n ");

    return 1;
}


/* inicializa el vector pasado como parametro con 1s */
int vector_initializacion_unos(float *v, int dim)
{
    for (int i = 0; i < dim; i++) {
        v[i] = 1.0;
    }

    return 1;
}