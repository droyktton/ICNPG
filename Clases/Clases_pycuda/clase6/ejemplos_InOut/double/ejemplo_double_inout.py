# -*- coding: utf-8 -*-
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
    
  __global__ void times_two(int N, float *a)
  {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
     
    if(id<N){
        a[id] = 2*a[id];
    }
  }
  
""")

N = 30000

a = np.ones(N)
a = a.astype(np.float32)

func = mod.get_function('times_two')

numThreads = 128
numBlocks = (N + numThreads - 1 )//numThreads

func(np.array(N), cuda.InOut(a), block=(numThreads,1,1), grid=(numBlocks,1,1))

print(a)

