# En cusp valen los mismos device backends que en thrust, y se usa igual.

# para compilar cuda usando nvcc 
# nvcc -o sagan sagan.cu -I /share/apps/icnpg/clases/common/

# para compilar openmp usando g++
# cp sagan.cu sagan.cpp
# g++ -O2 -o sagan sagan.cpp -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I /share/apps/icnpg/clases/common/ #-I /path/a/cuda/no/hace/falta/en/cluster

# para compilar serial cpu usando g++ (parece que anda con thrust pero no con cusp... si quiere cpu serial use openmp 1 thread)
# cp sagan.cu sagan.cpp
# g++ -O2 -o sagan sagan.cpp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I /share/apps/icnpg/clases/common/ #-I /path/a/cuda/no/hace/falta/en/cluster
