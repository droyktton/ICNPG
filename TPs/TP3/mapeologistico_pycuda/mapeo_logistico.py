# -*- coding: utf-8 -*-
# Mapeo logistico con cuda-kernel

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray


#----------------------------------------
# Parametros de entrada
#----------------------------------------

t_run = 1000
N = 100000
r_min = 2.4
r_max = 4.0

# creo dos eventos para la medicion del tiempo
start = cuda.Event()
end = cuda.Event()

# creacion de arrays en el host (numpy)   
r_host = np.linspace(r_min, r_max, N).astype(np.float32)
x_host = np.random.rand(N).astype(np.float32)
lambda_host = np.zeros_like(r_host) #para copiar los resultados al host, si es necesario

#-------------------------------------------------------------------
# METODO 1: Resolucion en CPU
#-------------------------------------------------------------------

def mapeo(t_run, r, x):
    acum = np.zeros(x.shape)
    for i in range(0, t_run):
        acum += np.log(np.abs(r - 2.0*r*x))
        x = r*x*(1.0 - x)
    l = acum/t_run
    return x, l

start.record() # timing con eventos de pycuda
start.synchronize() 
               
x_m01, lambda_m01 = mapeo(t_run, r_host, x_host)

end.record() # end timing
end.synchronize()
secs = start.time_till(end)*1e-3

print 'Metodo 01: %fs' %secs

#-------------------------------------------------------------------
# METODO 2: Resolucion con SourceModule
#-------------------------------------------------------------------

mod = SourceModule("""

  __global__ void mapeo()
  {
    
    /* TO DO: Complete el kernel de mapeo logistico, utilizando como argumentos 
        de entrada los vectores "x" y "r" y la cantidad de iteraciones "t_run". 
        Puede que tambien necesite pasar el largo "N" de los vectores.
        Guarde los resultados en los vectores "x" y "lambda".   
    */  
   
  }
""")

# Allocacion de arrays en el device y copia de datos:

# TO DO: Transifera los datos a la GPU. Puede hacerlo mediante la combinacion 
# de "cuda.mem_alloc()" y "cuda.memcpy_htod()" o mediante la utilizacion 
# de "cuda.In()", "cuda.Out()" o "cuda.InOut()".
# (también puede hacerlo con GPUarray mediante "gpuarray.to_gpu()")




# configuracion y llamado al kernel
mapeo_kernel = mod.get_function('mapeo')

numThreads = 128
numBlocks = (N + numThreads - 1 )/numThreads

start.record() # timing con eventos de pycuda

# TO DO: complete el llamado al kernel con los argumentos correspondientes.
# Recuerde que un entero "N" debe ser pasado como "np.array(N, dtype=np.int32)"
mapeo_kernel(block=(numThreads,1,1), grid=(numBlocks,1,1)) 

end.record() # end timing
end.synchronize()
secs = start.time_till(end)*1e-3


print 'Metodo 02: %fs' %secs

# TO DO: Copie los datos de vuelta al host en los arrays "x_m02", "lambda_m02",
# de ser necesario, con el metodo que corresponda.

#-------------------------------------------------------------------
# METODO 3: Resolucion con ElementWiseKernel 
#-------------------------------------------------------------------

# TO DO: Complete la definición del kernel elementwise utilizando como 
# argumentos de entrada los GPUarray correspondientes a "x" y "r". 
# Guarde los resultados en los GPUarray correspondientes a "x" y "lambda". 
# Puede optar por incluir el loop for dentro del kernel o hacerlo por fuera 
# en python (analice la performance de ambas opciones).

kernel = ElementwiseKernel( )

# TO DO: utilice GPUarray mediante "gpuarray.to_gpu()" para transferir los datos
# a la placa.


start.record() # start timing

# TO DO: complete el llamado al kernel con los argumentos correspondientes.
# Tenga en cuenta que este tipo de kernel no necesita que sea definido el tamaño 
# de bloque ni de grilla.
kernel()
   
    
end.record() # end timing
end.synchronize()
secs = start.time_till(end)*1e-3


print 'Metodo 03: %fs' %secs

# TO DO: Copie los datos de vuelta al host en los arrays "x_m03", "lambda_m03",
# utilizando el metodo ".get()" en el GPUarray.

#-------------------------------------------------------------------
# Guardar datos
#-------------------------------------------------------------------

# Opcional: Descomentando las siguientes lineas, puede guardar los datos en un 
# archivo .npz de numpy para luego graficarlos en su compuntadora 
# a traves de matplotlib (puede utilizar el script "graficar_mapeo.py" provisto)

# np.savez('salida_mapeo_m01.npz', r=r_host, x=x_m01, lamb = lambda_m01)
# np.savez('salida_mapeo_m02.npz', r=r_host, x=x_m02, lamb = lambda_m02)
# np.savez('salida_mapeo_m03.npz', r=r_host, x=x_m03, lamb = lambda_m03)