#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q cpu.q

## Pido N slots en una misma maquina, pruebe cambiar el numero...
#$ -pe neworte 14

## Limito los threads a cantidad de slots pedidos (podria largar mas threads...)
export OMP_NUM_THREADS=$NSLOTS

## para controlar nomas el numero de threads que la version omp va a usar
echo "numero de threads omp = " $OMP_NUM_THREADS

## nombre del nodo en el que cai...
hostname

##ejecuto el(los) binarios

## ejecuto version omp del codigo C
time ./laplace2d_omp

## o bien la version omp del codigo fortran 90
##time ./laplace2d_f90_omp
