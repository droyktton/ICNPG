1) Compilar:

$ nvcc programa.cu -o ejecutable

######################
2a) Ejecutar en su maquina con gpu

Sin argumentos:

$ ./ejecutable

Con argumentos:

$ ./ejecutable arg1 arg2 ...

######################
2b) Ejecutar en el Cluster:

$ qsub jobGPU 

donde jobGPU tiene que decir en su ultima linea, 
segun acepte o no argumentos

./ejecutable
./ejecutable arg1 arg2 ...


######################

3) El "make etc" no hace mas que automatizar la compilacion, la ejecucion u otras tareas.
Es solo por comodida. Que hace el make depende que haya en Makefile. 
En el Makefile que hemos preparado:

$ make 

hace el paso 1).

$ make run 

hace el paso 1) solo si es necesario y el 2a) con argumentos default (NTHX=5 y todas las otras en 1).

$ make run NTHY=2

lo mismo que antes 1)+2a) pero cambia el valor default de NTHY (se puede cambiar cualquiera de las 6).

$ make submit 

hace el paso 1) solo si es necesario y 2b) con argumentos default (NTHX=5 y todas las otras en 1).

$ make submit NTHY=2

lo mismo que antes, 1)+2b), pero cambia el valor default de NTHY (se puede cambiar cualquiera de las 6).

$ make submitwatch NTHY=2

hace lo mismo que el paso anterior, pero ademas, muestra el qstat actualiado cada 2s. Ctrl-C para salir.

######################








