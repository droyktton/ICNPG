import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.cumath as cumath

####################################################
### Elementwise Functions on GPUArray Instances ####
####################################################

a_gpu = gpuarray.GPUArray([50,60], dtype = np.float32)
a_gpu.fill(144.0)

i_gpu = cumath.fabs(a_gpu)

j_gpu = cumath.floor(a_gpu)

k_gpu = cumath.exp(a_gpu)

l_gpu = cumath.log(a_gpu)

m_gpu = cumath.sqrt(a_gpu)

p_gpu = cumath.sin(a_gpu)

q_gpu = cumath.tan(a_gpu)

print(m_gpu)

