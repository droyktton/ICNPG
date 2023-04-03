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
module load cuda-6.5 
#ejecuto el binario elegir dimensiones correctas
/usr/local/cuda/bin/nvprof ./main 256 32 
