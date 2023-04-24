#@title en gpu con cupy pero de mas bajo nivel,usando SIMPLECUDA kernel 
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


# Define matrix sizes
m = my_integer
n = my_integer
p = my_integer
k = my_integer

device = cp.cuda.Device()

# Create random matrices on the GPU
a_gpu = cp.random.rand(m, n).astype(cp.float32)
b_gpu = cp.random.rand(n, k).astype(cp.float32)
c_gpu = cp.zeros((m, k)).astype(cp.float32)

# Define the kernel code as a string
kernel_code = """
extern "C" __global__ void matrix_multiply(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
"""

# Compile the kernel code into a function
matrix_multiply = cp.RawKernel(kernel_code, 'matrix_multiply')

# Define the block and grid sizes for the kernel
block_size = (16, 16, 1)
grid_size = ((m + block_size[0] - 1) // block_size[0], (k + block_size[1] - 1) // block_size[1], 1)


# warming up
matrix_multiply(grid_size, block_size, (a_gpu, b_gpu, c_gpu, m, n, k))

start_time = time.time()

# Call the kernel function with the input matrices and sizes
matrix_multiply(grid_size, block_size, (a_gpu, b_gpu, c_gpu, m, n, k))

device.synchronize()
end_time = time.time()

# Print the elapsed time

dev=cp.cuda.runtime.getDevice()
props=cp.cuda.runtime.getDeviceProperties(dev)

print("N=",n ,", Elapsed time: ", (end_time - start_time)*1000, "mseconds", " in ", props['name'])


# Copy the result matrix back to the CPU and print it
c_cpu = c_gpu.get()
#print(c_cpu)

