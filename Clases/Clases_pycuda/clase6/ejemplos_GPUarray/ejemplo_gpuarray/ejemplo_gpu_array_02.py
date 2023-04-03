import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

########################################
#### Constructing GPUArray Instance ####
########################################

a_gpu = gpuarray.GPUArray([50,60], dtype = np.float32)
a_gpu.fill(3.14)

a = np.ones([50,60]).astype(np.float32)*3.14
a_gpu = gpuarray.to_gpu(a) #<- copia a GPU

a_gpu = gpuarray.empty([50,60], dtype = np.float32)
a_gpu = gpuarray.zeros([50,60], dtype = np.float32)
# d_gpu = gpuarray.ones([100,120], dtype = np.float32) # <- no existe

b_gpu = gpuarray.empty_like(a_gpu)
b_gpu = gpuarray.zeros_like(a_gpu)
b_gpu = gpuarray.ones_like(a_gpu)

c_gpu = gpuarray.arange(0, 60, dtype = np.float32)

print(c_gpu)