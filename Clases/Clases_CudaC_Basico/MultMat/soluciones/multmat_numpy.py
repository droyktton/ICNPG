#@title multipliquemos dos matrices de 4096x4096 con numpy (cpu)

import numpy as np
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

# Generate random matrices with the given sizes
A = np.random.rand(m, n)
B = np.random.rand(n, p)

# Compute the matrix product with a timer
start_time = time.time()
C = np.dot(A, B)
end_time = time.time()


# Print the elapsed time
print("Elapsed time: ", (end_time - start_time)*1000, "mseconds")
