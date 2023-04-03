#! /bin/bash
#   El path de ejecucion del job es el directorio actual
#$ -cwd
#   Reune stdout y stderr en .o666
#$ -j y
#   Bourne shell para el job
#$ -S /bin/bash
#   Nombre del job
#$ -N HolaMundo
#   pido la cola cpu.q
#$ -q cpu.q
#
# imprime el nombre del nodo
hostname
#ejecuto el binario
./main
