import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
#include<stdio.h>
    __global__ void Quiensoy()
    {
    	printf("Soy el thread (%d,%d,%d) del bloque (%d,%d,%d) [blockDim=(%d,%d,%d),gridDim=(%d,%d,%d)]\\n",
     	threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,
		blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);

    }
    """)


quien = mod.get_function("Quiensoy")

quien(block=(6,1,1), grid=(4,1,1))

