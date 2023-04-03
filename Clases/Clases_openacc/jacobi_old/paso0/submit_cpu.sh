#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash

## pido cola de cpu
#$ -q cpu.q

## nombre del nodo en el que cai...
hostname

##ejecuto el(los) binarios

## ejecuto version omp del codigo C
time ./laplace2d

