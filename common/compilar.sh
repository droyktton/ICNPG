#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: compilar.sh <filename>"
    exit 1
fi

filename=$1

nvcc -I /state/partition1/apps/spack/opt/spack/linux-centos7-x86_64/gcc-11.1.0/eigen-3.4.0-xaed56heqf6sisylasau4wt7dvmbgp2r/include/eigen3/ "$filename" -lcusparse -lcusolver -lcublas -std=c++11  -o CHOL -DCHOL
nvcc -I /state/partition1/apps/spack/opt/spack/linux-centos7-x86_64/gcc-11.1.0/eigen-3.4.0-xaed56heqf6sisylasau4wt7dvmbgp2r/include/eigen3/ "$filename" -lcusparse -lcusolver -lcublas -std=c++11  -o QR -DQR
nvcc -I /state/partition1/apps/spack/opt/spack/linux-centos7-x86_64/gcc-11.1.0/eigen-3.4.0-xaed56heqf6sisylasau4wt7dvmbgp2r/include/eigen3/ "$filename" -lcusparse -lcusolver -lcublas -std=c++11  -o LU -DLU
