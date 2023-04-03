le agrego openacc directives a la convolucion secuencial de moni para que corra en gpu
una aceleracion 5x (*) se ve con la mas simple de las directivas, #pragma acc kernels

(*) el timming que hace incluye las copias D-H-D, no como el ejemplo de clase 0


El Makefile tiene la compilacion con openacc y openmp juntas. 
Haciendo make se generan tres ejecutables, uno para gpu y dos para multicore usando pgc++ y g++. 

Si no quiere usar Makefile, compile asi:
-------
OPENACC

pgc++ -acc -Minfo fuente.c -o ejecutable

sacar el -acc para compilar normal, secuencial, con posibles optimizaciones...

------
OMP:

pgc++ -fast -mp -Minfo fuente.x -o ejecutable

si no quiere usar pgc++ (como no tenemos licencia premium max threads=4) puede usar g++

gcc -O2  -fopenmp -lgomp -o convolucion1d_gnuomp convolucion_openacc.cpp
------

Para ver en funcion del numero de omp threads podemos hacer

for n in 1 2 3 4; do export OMP_NUM_THREADS=$n; echo $n $(./convolucion1d_omp); done
