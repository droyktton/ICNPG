import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
N = 1000000

#data = cp.random.normal(size=N)
data = cp.random.uniform(low=0.0, high=1.0, size=(N), dtype=float)

# Compute the histogram
num_bins = 10
hist, bins = cp.histogram(data, bins=num_bins)

h=hist.get()
b=bins.get()

# Combine the vectors into a 2D array with two columns
data = np.column_stack((b[:-1], h))

# Save the array to a file with a space delimiter
np.savetxt('histo.txt', data, delimiter=' ', fmt='%.2f')

# Plot the histogram
#plt.bar(b[:-1], h, width=(b[1]-b[0]), align='edge')

# Add title and labels
#plt.title('cupy')
#plt.xlabel('valores')
#plt.ylabel('conteo')

# Show the plot
#plt.show()

