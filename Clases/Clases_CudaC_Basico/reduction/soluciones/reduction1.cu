#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <ctime>
#include <sys/time.h>
#include <sstream>
#include <string>
#include <fstream>

using namespace std;

__global__ void reduce3(int *g_idata, int *g_odata, int size)
{
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

   for(unsigned int s=blockDim.x*gridDim.x/2; s>0; s>>=1)
   { 
        if (i < s) {
         g_idata[i] += g_idata[i + s];
        }
   }

   if (i == 0) g_odata[blockIdx.x] = g_idata[blockIdx.x*blockDim.x];
}

__global__ void reduce2(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

   for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
	if (tid < s) {
		sdata[tid] += sdata[tid + s];
	}
	__syncthreads();
   }
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}





__global__ void reduce0(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

  for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
         sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
     }

   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce1(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

   for(unsigned int s=1; s < blockDim.x; s *= 2) 
   {
	int index = 2 * s * tid;
	if (index < blockDim.x) {
		sdata[index] += sdata[index + s];
	}
	__syncthreads();
   }

   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}




int main(int argc, char **argv){

  int size = atoi(argv[1]);
  // crea un vector de host de size "ints" y lo inicializa a 1
  thrust::host_vector<int> data_h_i(size, 1);

  //initialize the data, all values will be 1
  //so the final sum will be equal to size

  int threadsPerBlock = 256;
  int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;

  // una forma de "empaquetar" vectores de device usando thrust
  // crea y aloca un vector de device de "int" y copia el contenido de un vector de host 
  thrust::device_vector<int> data_v_i = data_h_i;

  // crea y aloca un vector de device de totalBlocks elementos "int" 
  thrust::device_vector<int> data_v_o(totalBlocks);

  // los vectors de device son algo mas que un punter a memoria de device
  int* output = thrust::raw_pointer_cast(data_v_o.data());
  int* input = thrust::raw_pointer_cast(data_v_i.data());

  switch(atoi(argv[2])){
	case 2:
	std::cout << "reduction 2\n";
  	reduce2<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
  	reduce2<<<1, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, totalBlocks);
	break;

	case 3:
	std::cout << "reduction 3\n";
  	reduce3<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
	break;

	case 0:
	std::cout << "reduction 0\n";
  	reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
  	reduce0<<<1, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, totalBlocks);
	break;

	case 1:
	std::cout << "reduction 1\n";
  	reduce1<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
  	reduce1<<<1, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, totalBlocks);
	break;


	default:
	std::cout << "reduction 0\n";
  	reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
  	reduce0<<<1, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, totalBlocks);
  }

  // todo lo anterior se puede hacer en una línea usando la biblioteca thrust 
  //cout << "con thrust: " << thrust::reduce(thrust::device,input,input+size) << endl;

  data_v_o[0] = data_v_i[0];
  data_v_i.clear();
  data_v_i.shrink_to_fit();

  thrust::host_vector<int> data_h_o = data_v_o;

  data_v_o.clear();
  data_v_o.shrink_to_fit();

  cout<< "con CUDA C: " << data_h_o[0]<<endl;

  return 0;

}
