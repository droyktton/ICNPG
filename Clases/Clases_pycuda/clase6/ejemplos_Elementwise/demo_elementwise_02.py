from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.curandom import rand as curand
import numpy as np
import pycuda.cumath as cumath


lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *y, float *z",
        "z[i] = my_f(a*x[i], b*y[i])",
        "linear_combination",
        preamble="""
        __device__ float my_f(float x, float y)
        { 
          return sin(x*y);
        }
        """)

a_gpu = curand((50,))
b_gpu = curand((50,))

c_gpu = gpuarray.empty_like(a_gpu)
lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

print(np.all((c_gpu - cumath.sin(5*a_gpu*6*b_gpu)).get() < 1e-5))
