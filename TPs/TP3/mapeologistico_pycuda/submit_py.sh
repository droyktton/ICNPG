#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
##  pido la cola gpu.q
#$ -q gpu.q@compute-0-1
#
module load cuda-7.5 opt-python

#ejecuto el script
python mapeo_logistico.py
