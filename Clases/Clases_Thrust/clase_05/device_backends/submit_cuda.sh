#! /bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpushort
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
#cargar variables de entorno para encontrar cuda

module load cuda/10.0.130

#ejecutar el o los binarios con sus respectivos argumentos


echo "========="
hostname
time ./cuda.out
echo "========="
