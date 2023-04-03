import numpy as np
import matplotlib.pyplot as plt

output_m01 = np.load('salida_mapeo_m01.npz')
output_m02 = np.load('salida_mapeo_m02.npz')
output_m03 = np.load('salida_mapeo_m03.npz')

r_host = output_m01['r']
x_m01 = output_m01['x']
lambda_m01 = output_m01['lamb']

x_m02 = output_m02['x']
lambda_m02 = output_m02['lamb']

x_m03 = output_m03['x']
lambda_m03 = output_m03['lamb']

# graficos en matplotlib
plt.figure()
plt.plot(r_host, lambda_m01,'r-')
plt.plot(r_host, lambda_m02,'g-')
plt.plot(r_host, lambda_m03,'b-')
plt.show()

plt.figure()
plt.plot(r_host, x_m01,'r.')
plt.plot(r_host, x_m02,'g.')
plt.plot(r_host, x_m03,'b.')
plt.show()

