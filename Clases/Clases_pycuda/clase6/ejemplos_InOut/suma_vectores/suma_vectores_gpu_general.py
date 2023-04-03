# solucion paralela optima...

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import sys
import numpy as np

SIZE = 1024

mod = SourceModule("""

// kernel
__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	// indice de thread mapeado a indice de array 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

""")

# Pasaje de argumentos por linea de comando
if len(sys.argv)==2:
	N = int(sys.argv[1]);
else:
	N=SIZE;

print(N)

# create two timers so we can speed-test each approach
start = cuda.Event()
end = cuda.Event()

# inicializacion arrays de host
a = np.arange(N, dtype = np.int32)
b = np.arange(N, dtype = np.int32)
c = np.zeros(N, dtype = np.int32)

# alocacion memoria de device
# d_a = cuda.mem_alloc(a.nbytes)
# d_b = cuda.mem_alloc(b.nbytes)
# d_c = cuda.mem_alloc(c.nbytes)


# 	// copia de host a device
# cuda.memcpy_htod(d_a, a)
# cuda.memcpy_htod(d_b, b)
# cuda.memcpy_htod(d_c, c)

vector_add = mod.get_function('VectorAdd')
# timer para gpu...
start.record() # start timing

# grilla de threads suficientemente grande...
nThreads = 256
nBlocks = int((N + nThreads - 1 )/nThreads)

# suma paralela en el device
vector_add(cuda.In(a), cuda.In(b), cuda.Out(c), np.array(N), block=(nThreads,1,1), grid=(nBlocks,1,1))
end.record() # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3

print("GPU time:")
print("%fs\n" % secs)

# 	copia (solo del resultado) del device a host
# cuda.memcpy_dtoh(c, d_c)

# verificacion del resultado
estan_cerca = np.allclose(c, a+b, atol =0.0000001) 
print("Estan cerca?", estan_cerca)

print(a)

