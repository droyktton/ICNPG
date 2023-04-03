echo "OMP with pgc++ (CPU)"
for((n=1;n<5;n++)); do export OMP_NUM_THREADS=$n; echo $n $(./convolucion1d_omp); done

echo
echo "OMP with g++ (CPU)"
for((n=1;n<5;n++)); do export OMP_NUM_THREADS=$n; echo $n $(./convolucion1d_gnuomp); done

echo 
echo "OPENACC (GPU)"
./convolucion1d_acc
