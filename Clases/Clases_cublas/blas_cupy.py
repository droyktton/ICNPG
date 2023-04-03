import cupy as cp 
import numpy as np 
import time
from timeit import default_timer as timer

a_gpu = cp.random.rand(1024,1024).astype(np.float32)
b_gpu = cp.random.rand(1024,1024).astype(np.float32)

c_gpu = cp.matmul(a_gpu,b_gpu,out=None)

iteraciones=100

t1 = timer()
for x in range(iteraciones):
	c_gpu = cp.matmul(a_gpu,b_gpu,out=None)
cp.cuda.Stream.null.synchronize()
#cp.cuda.Device(0).synchronize()
t2 = timer()
totalcupy=(t2-t1)

a_cpu = cp.asnumpy(a_gpu)
b_cpu = cp.asnumpy(b_gpu)

t3 = timer()
for x in range(iteraciones):
	c_cpu = np.matmul(a_cpu,b_cpu,out=None)
t4 = timer()
totalnumpy=(t4-t3)

#print("\na_cpu: ", a_cpu) 
#print("\nb_cpu: ", b_cpu) 
#print("\nResults:")
#print("\nc_gpu: ", c_gpu, " in ", totalcupy, " s") 
#print("\nc_cpu: ", c_cpu, " in ", totalnumpy, " s") 

print("tgpu={}".format(totalcupy)) 
print("tcpu={}".format(totalnumpy)) 


