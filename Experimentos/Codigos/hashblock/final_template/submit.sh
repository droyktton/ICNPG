#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash

##  pido la cola gpu.q
#$ -q gpushort.q

## pido una placa
#$ -l gpu=1

## nombre del job
#$ -N final2
#

## ejecutables (elija el block que quiera)
./a.out blocks/block.471972
