/* 
   Guia 1, Ej 1
   ICNPG 2016
 
   Complete el template para que arroje la informacion requerida de la placa.
*/

#include <stdio.h>
#include <cuda_runtime_api.h>
int *pArgc = NULL;
char **pArgv = NULL;

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {

        cudaDeviceProp deviceProp;

	// elije device dev
	cudaSetDevice(dev);

	// carga las propiedades de device dev en deviceProp
        cudaGetDeviceProperties(&deviceProp, dev);

	//// Incluir el código aquí.

    }

}


















