file="multmat_solucion.cu"
nvcc $file -lcublas -DCUBLASXt -o cublasxt
nvcc $file -lcublas -DCUBLASN -o cublas
nvcc $file -lcublas -DCPU -o naivecpu
nvcc $file -lcublas -DSIMPLECUDA -o naivecuda
