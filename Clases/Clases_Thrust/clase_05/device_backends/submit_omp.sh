#! /bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q cpuINgpu
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
#cargar variables de entorno para encontrar cuda

module load cuda/10.0.130

#ejecutar el o los binarios con sus respectivos argumentos

## Pido N slots en una misma maquina, pruebe cambiar el numero...
#$ -pe neworte 14

## Limito los threads a cantidad de slots pedidos
export OMP_NUM_THREADS=$NSLOTS

#ejecuto los binarios

echo "=================="
echo "numero de threads omp = " $OMP_NUM_THREADS
hostname
time ./omp.out
echo "=================="
