/*
Fuentes: ising.cpp, util_CPU.h

Este programa simula el modelo de Ising en dos dimensiones en el HOST en una matriz de LxL sitios a temperatura T.
https://en.wikipedia.org/wiki/Ising_model (sin campo externo y con J=1).

Se usa el algoritmo de metropolis, con updates sincronicos de los sitios con la misma paridad.
Esta forma de iterar pone en evidencia una posibilidad de paralizacion muy sencilla, primero 
sobre todas los casilleros negros (sitios pares), y luego sobre los blancos (sitios impares), 
como si fuera un tablero de ajedrez.  

Para compilar la version de HOST que proveemos:

g++ ising.cpp -o ising_cpu

Para correr:

./ising_cpu 128 2.0 10000 1

donde por ejemplo hemos usado L=128, T=2.0, iteraciones totales trun=10000 y semilla global seed=1

OUTPUT NORMAL
Genera un archivo "magnetizacion.dat", conteniendo la magnetizacion en funcion del tiempo.
(magnetizacion = igual a la suma de las variables de sitio/numero total de sitios) 

ANIMACION (OPCIONAL)
Si quiere hacer una animacion, tomando snapshot cada X iteraciones haga lo siguiente:

g++ ising.cpp -o ising_cpu -DMOVIE=X

donde hemos definido el macro MOVIE que se iguala al numero de iteraciones entre frames de la pelicula.
Corra el programa como siempre, por ejemplo asi:

./ising_cpu 128 2.0 1000 1

Esto generara un archivo "evolucion.dat" (ojo que puede ser muy grande si L>>1 y trun/X >> 1).
Si quiere visualizar en forma simple corra por ejemplo en gnuplot:

gnuplot> set term gif animate; set out "movie.gif"; i=0;iend=500; L=128; load 'ver.gnu'

para generar una pelicula gif animada con, por ejemplo, 500 cuadros para un sistema de tamanio L=128. 
El archivo "ver.gnu" esta provisto. Modifiquelo a piacere, o use su graficador preferido.

=============================================================================
TODO: 

1) CODIGO: 
Paralelice este programa para que corra en GPU. 
Para ello primero entienda el problema y el codigo de GPU. Compilelo y corralo.
Luego escriba su propio "util_GPU.h" modificando las funciones de "util_CPU.h" y 
agregando lo que considere necesario. Use cualquiera de las herramientas y 
librerias vistas en clase, pero no modifique la logica del programa (adaptese a ella).  

No cambie la funcion main(). Puede cambiar, si necesita (pero no es realmente necesario), 
el tipo de argumentos de las funciones de HOST dadas en "util_CPU.h", pero no cambie 
la logica. Cada funcion debe realizar la misma tarea pero en DEVICE. 
Las funciones de device escribalas en un archivo "util_GPU.h" que reemplazara 
a "util_CPU.h" cuando compile para GPU. 

2) PRUEBE LA CORRECTITUD FISICA DEL PROGRAMA:

Muestre graficamente que, siendo Tc=2/log(1+sqrt(2)): 

a* Para T<Tc la magnetizacion converge a un valor que fluctua alrededor de un valor finito (negativo o positivo).

b* Para T>Tc la magnetizacion converge a un valor que fluctua alrededor de zero.

c* Para N>>1 a la temperatura Tc=2/log(1+sqrt(2)) la relajacion de la magnetizacion 
desde un estado random hacia su valor de equilibrio es la mas lenta (compare con T<Tc y T>Tc). 

d* Presente tres "snapshots" de la matriz para N>>1, trun>>1, exactamente a la temperatura Tc, para T<Tc y para T>Tc. 

[Use estos criterios fisicos para testear la correctitud del codigo de DEVICE y de HOST.]

3) Compare la performance del codigo de DEVICE que escriba con la del HOST provista, graficando tiempo de calculo 
de CPU (de solo un thread) y de GPU en funcion de del numero de sitios N, para N>>1. Haga el grafico para al menos 
dos GPUs diferentes. Discuta y explique la aceleracion obtenida.


Si no entiende algo del enunciado, haganoslo saber a la brevedad. Intentaremos responder a la idem.

*/

=============================================================================

Para compilar la version experimental

g++ ising_experimental.cpp -O2 -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP \
-lgomp -I /usr/local/cuda-5.5/include/

esta incluye deteccion de clusters usando cusp connected components algorithm.

==============================================================================

Solucion del parcial: ising.cu util_GPU.h
