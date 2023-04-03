#include <stdio.h>
#include <stdlib.h>


// kernel
__global__ void Quiensoy()
{
	printf("Soy el thread (%d,%d,%d) del bloque (%d,%d,%d) [blockDim=(%d,%d,%d),gridDim=(%d,%d,%d)] \n",threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,
blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);

}

int main(int argc, char **argv)
{

	if(argc!=7) {
		// ejemplo 1
		printf("ejemplo 1:\n");
		Quiensoy<<< 3, 2>>>();
		cudaDeviceSynchronize();
	
		// ejemplo 2
		printf("ejemplo 2:\n");
		dim3 nb(2,2); 
		dim3 nt(2,1);
		Quiensoy<<< nb, nt >>>();
		cudaDeviceSynchronize();	
	}	
	else{
		//printf("uso: %s ntbx ntby ntbz nbgx nbgy nbgz\n", argv[0]);
		dim3 nThreads_per_block;
		dim3 nBlocks_per_grid;

		nThreads_per_block.x = atoi(argv[1]);
		nThreads_per_block.y = atoi(argv[2]);
		nThreads_per_block.z = atoi(argv[3]);
		nBlocks_per_grid.x = atoi(argv[4]);
		nBlocks_per_grid.y = atoi(argv[5]);
		nBlocks_per_grid.z = atoi(argv[6]);

		// kernel
		Quiensoy<<< nBlocks_per_grid, nThreads_per_block >>>();
		cudaDeviceSynchronize();	
	}
	return 0;
}
