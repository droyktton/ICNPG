#include <nvToolsExt.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/reduce.h>
#include<thrust/fill.h>


__global__ void reduce_mala(float *g_idata, float *g_odata, int size){

   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   if(i<size) atomicAdd(&g_odata[0],g_idata[i]);
}

// block reduction
__global__ void reduce(float *g_idata, float *g_odata, int size){

   extern __shared__ float sdata[];

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
   if (tid == 0) 
   // antes guardabamos los resultados parciales
   //g_odata[blockIdx.x] = sdata[0];
   // ahora sumamos todo en el mismo kernel
   atomicAdd(&g_odata[0],sdata[0]);
}

int main(){
  int N=1000000;
 
  thrust::device_vector<float> input(N,1.f/N);
  thrust::device_vector<float> output(N,0.0f);

  int threads=256;
  int blocks=(N+threads-1)/threads;

  float *inputraw=thrust::raw_pointer_cast(input.data());
  float *outputraw=thrust::raw_pointer_cast(output.data());

  /////
  nvtxRangePush("-----block+atomic reduce-----");

  size_t mem=threads*sizeof(float);
  reduce<<<blocks,threads,mem>>>(inputraw,outputraw,N);
  float res1=output[0];
  cudaDeviceSynchronize();

  nvtxRangePop();
  /////


  thrust::fill(input.begin(),input.end(),1.f/N);
  thrust::fill(output.begin(),output.end(),0.f);

  /////
  nvtxRangePush("-----atomic reduce-----");

  reduce_mala<<<blocks,threads>>>(inputraw,outputraw,N);
  float res2=output[0];
  cudaDeviceSynchronize();

  nvtxRangePop();
  /////

  thrust::fill(input.begin(),input.end(),1.f/N);
  thrust::fill(output.begin(),output.end(),0.f);

  /////
  nvtxRangePush("-----thrust reduce-----");

  float res3=thrust::reduce(input.begin(),input.end());
  cudaDeviceSynchronize();

  nvtxRangePop();
  /////

  thrust::fill(input.begin(),input.end(),1.f/N);
  thrust::fill(output.begin(),output.end(),0.f);
  thrust::host_vector<float> hinp(input);
  thrust::host_vector<float> hout(output);

  /////
  nvtxRangePush("-----thrust host reduce-----");

  float res4=thrust::reduce(hinp.begin(),hinp.end());
  
  // seria lo mismo que esto:
  /*float res4=0.0;
  for(int i=0;i<N;i++){
    res4+=hinp[i];
  }*/

  nvtxRangePop();
  /////


  std::cout << res1 << " " << res2 << " " << res3 << " " << res4 << std::endl;

  return 0;
}

