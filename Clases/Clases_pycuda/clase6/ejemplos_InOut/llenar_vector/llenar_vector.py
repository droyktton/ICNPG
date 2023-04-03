import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import sys
import numpy as np

SIZE = 1024

mod = SourceModule("""

	// funcion para rellenar
	__host__ __device__ float Mifuncion(int i)
	{
		return tanh(cos(exp(-i*0.01)+0.02));
	}


	// kernel para tabular
	__global__ void Llenar(float *a)
	{
		// indice de thread mapeado a indice de array 
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		a[i]=Mifuncion(i);
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

a = np.empty(N)
a = a.astype(np.float32)

# d_a = cuda.mem_alloc(a.nbytes)


llenar = mod.get_function('Llenar')

# grilla de threads suficientemente grande...
nThreads = 256
nBlocks = int((N + nThreads - 1 )/nThreads)

# timer para gpu...
start.record() # start timing

# llena en el device
llenar(cuda.InOut(a), block=(nThreads,1,1), grid=(nBlocks,1,1))
end.record() # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3

print("GPU time:")
print("%fs\n" % secs)

# copia (solo del resultado) del device a host
# cuda.memcpy_dtoh(a, d_a)

# En numpy

b = np.arange(N).astype(np.float32)

# timer para cpu...
start.record() # start timing
start.synchronize()

b = np.tanh(np.cos(np.exp(-b*0.01)+0.02))

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("CPU time:")
print("%fs\n" % secs)

estan_cerca = np.allclose(a, b, atol =0.0000001) 
print("Estan cerca?", estan_cerca)
