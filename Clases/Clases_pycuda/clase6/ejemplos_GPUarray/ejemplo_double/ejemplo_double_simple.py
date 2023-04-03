import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np

N = 30000

a = np.ones(N)
a = a.astype(np.float32)

a_gpu = gpuarray.to_gpu(a)
a_doubled = (2*a_gpu).get()

print(a)
print(a_doubled)