APPS=cublasxt cublas naivecpu naivecuda

all: ${APPS}

cublasxt: multmat_solucion.cu
	nvcc multmat_solucion.cu -lcublas -DCUBLASXt -o cublasxt

cublas: multmat_solucion.cu
	nvcc multmat_solucion.cu -lcublas -DCUBLAS -o cublas

naivecpu: multmat_solucion.cu
	nvcc multmat_solucion.cu -lcublas -DSIMPLECPU -o naivecpu

naivecuda: multmat_solucion.cu
	nvcc multmat_solucion.cu -lcublas -DSIMPLECUDA -o naivecuda

clean:
	rm -f ${APPS}

submit:	all
	qsub jobGPU; watch qstat
