import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np


my_gpu_array = gpuarray.GPUArray([5,5], dtype = np.float32)


print(my_gpu_array)
print(my_gpu_array.dtype)
print(my_gpu_array.shape)
print(my_gpu_array.size)

print(my_gpu_array.nbytes)
print(my_gpu_array.strides)

print(my_gpu_array.ptr)


