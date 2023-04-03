import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.reduction import ReductionKernel
import numpy as np

a = gpuarray.arange(400, dtype=np.float32)
b = gpuarray.arange(400, dtype=np.float32)

krnl = ReductionKernel(np.float32, neutral="0",
			reduce_expr="a+b", map_expr= "x[i]*y[i]",
			arguments="float *x, float *y")

my_dot_prod = krnl(a, b).get()

print(my_dot_prod)
print( (my_dot_prod - gpuarray.sum(a*b).get()) < 1e-6)
