import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

###################
### Reductions ####
###################
a_gpu = gpuarray.GPUArray([50,60], dtype = np.float32)
a_gpu.fill(3.14)

b_gpu = gpuarray.arange(0, 3000, dtype = np.float32)


print(gpuarray.sum(a_gpu))

print(gpuarray.dot(a_gpu, b_gpu))

print(gpuarray.max(b_gpu))

print(gpuarray.min(b_gpu))


