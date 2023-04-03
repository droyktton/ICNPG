#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola gpu.q
#$ -q gpu.q
## pido una placa
#$ -l gpu=1
#

## para saber en que placa, usar cuda o openacc api desde el codigo...
echo "en alguna GPU..."

## imprime nombre del nodo
hostname

##ejecuto el(los) binario(s)

## corre version openacc del codigo C
time ./laplace2d_acc

## corre version openacc del codigo fortran90
## time ./laplace2d_f90_acc
