/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>    // std::shuffle
#include <iostream>
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include "gpu_timer.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


// incremento con indices permutados
template <typename T>
__global__ void miKernel(T* a, int *ind)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = ind[i];
  a[j] = a[j] + 1;
}


template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;
  float ms;

  int n = nMB*1024*1024/sizeof(T);

  thrust::device_vector<T> d_a(n);
  T *rawd_a= thrust::raw_pointer_cast(d_a.data()); 

  thrust::device_vector<int> d_indices(n);
  int *rawindices = thrust::raw_pointer_cast(d_indices.data()); 

  gpu_timer cronometro;
     
//////////////////////////////////
  printf("\n\n#acceso coherente: Bandwidth (GB/s), primetos 5 indices:\n");
 
  miKernel<<<n/blockSize, blockSize>>>(rawd_a, rawindices); // warm up

  // inicializacion de arrays
  thrust::sequence(d_indices.begin(),d_indices.end(),0);  //d_indices={0,1,2,...,n-1}

  thrust::fill(d_a.begin(),d_a.end(),T(0.0));	  	  //d_a={0,0,...,0}

  cronometro.tic();
  miKernel<<<n/blockSize, blockSize>>>(rawd_a, rawindices); 
  ms=cronometro.tac();
 
  std::cout << 2*nMB/ms << " [";
  for(int i=0;i<5;i++) std::cout << d_indices[i] << ",";
  std::cout << "...]" << std::endl;
  
///////////////////////////////////////  
  std::srand ( unsigned ( std::time(0) ) );

  // inicializacion 
  // creamos un array de indices permutado aleatoriamente
  thrust::host_vector<int> h_indices(n);
  thrust::sequence(h_indices.begin(),h_indices.end()); 
  
  printf("\n\n#acceso random: Bandwidth (GB/s), primetos 5 indices:\n"); 
  for(int p=0;p<32;p++){

  	  std::random_shuffle (h_indices.begin(), h_indices.end());
	  thrust::copy(h_indices.begin(),h_indices.end(),d_indices.begin()); // d_indices={3,4,0,n-1,2,...}

          thrust::fill(d_a.begin(),d_a.end(),T(0.0));//d_a={0,0,...,0}

  	  cronometro.tic();
	  miKernel<<<n/blockSize, blockSize>>>(rawd_a, rawindices); 
  	  ms=cronometro.tac();
  	  
	  std::cout << 2*nMB/ms << " [";
	  for(int i=0;i<5;i++) std::cout << d_indices[i] << ",";
	  std::cout << "...]" << std::endl;

   }
    
////////////////////////////////////////
  thrust::sequence(h_indices.begin(),h_indices.end(),0); 
  float ms_sort;

  printf("\n\n#acceso random recauchutado: Bandwidth (GB/s), primetos 5 indices:\n"); 
  for(int p=0;p<32;p++){
  	  std::random_shuffle (h_indices.begin(), h_indices.end());
	  thrust::copy(h_indices.begin(),h_indices.end(),d_indices.begin()); // d_indices={3,4,0,n-1,2,...}

          thrust::fill(d_a.begin(),d_a.end(),T(0.0));//d_a={0,0,...,0}

  	  cronometro.tic();
  	  thrust::sort_by_key(d_indices.begin(),d_indices.end(),d_a.begin()); 
  	  ms_sort=cronometro.tac();

  	  cronometro.tic();
	  miKernel<<<n/blockSize, blockSize>>>(rawd_a, rawindices); 
  	  ms=cronometro.tac();
  	  
	  std::cout << 2*nMB/ms << ", pero con sort " << 2*nMB/ms_sort << " [";
	  for(int i=0;i<5;i++) std::cout << d_indices[i] << ",";
	  std::cout << "...]" << std::endl;
   }
    
}

int main(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId) )
  ;
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("#Device: %s\n", prop.name);
  printf("#Transfer size (MB): %d\n", nMB);
  
  printf("#%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}
