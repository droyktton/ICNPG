#compilacion serial con NVCC o g++ (elegir)
#nvcc -Xcompiler -fopenmp -lgomp device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -o cpp.out
cp device_backends.cu device_backends.cpp
g++ -fopenmp -lgomp device_backends.cpp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -o cpp.out


#compilacion paralela openmp con NVCC o g++
#nvcc -Xcompiler -fopenmp -lgomp device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -o omp.out
cp device_backends.cu device_backends.cpp
g++ -fopenmp -lgomp device_backends.cpp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -o omp.out

#compilacion normal cuda con NVCC (solo con nvcc)
nvcc device_backends.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -o cuda.out

qsub submit_cuda.sh
qsub submit_omp.sh
qsub submit_cpp.sh

#./cpp.out
#./cuda.out
#./omp.out
