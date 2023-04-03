#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola gpu.q
#$ -q gpushort.q
## pido una placa
#$ -l gpu=1
#
#ejecuto el binario

#/usr/local/cuda-5.5/bin/nvprof ./simple_cufft
./Brown_CUDA
