#@title Ahora lo hacemos en gpu con cupy

import cupy as cp
import time
import sys

# Check if there are at least 2 arguments
if len(sys.argv) < 2:
    print("Usage: python myscript.py <integer>")
    sys.exit(1)

# Get the second argument and convert it to an integer
try:
    my_integer = int(sys.argv[1])
except ValueError:
    print("The argument should be an integer")
    sys.exit(1)


# Set the sizes of the matrices
m = my_integer
n = my_integer
p = my_integer

device = cp.cuda.Device()

# Generate random matrices with the given sizes
A = cp.random.rand(m, n)
B = cp.random.rand(n, p)

# Compute the matrix product with a timer
C = cp.dot(A, B) #warmup!
start_time = time.time()
C = cp.dot(A, B)
device.synchronize()
end_time = time.time()

# Print the elapsed time

dev=cp.cuda.runtime.getDevice()
props=cp.cuda.runtime.getDeviceProperties(dev)

print("N=",n ,", Elapsed time: ", (end_time - start_time)*1000, "mseconds", " in ", props['name'])


