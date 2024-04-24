APPS=convolucion_solucion

all: ${APPS}

INCLUDES= -I/share/apps/icnpg/common

%: %.cu
	nvcc ${INCLUDES} -o $@ $<

clean:
	rm -f ${APPS} *.dat

run:	
	./convolucion

submit:	
	qsub jobGPU; watch qstat



#define SIMPLECPU
#define SIMPLECUDA
#define CUBLAS
#define CUBLASXt
#file="multmat_solucion.cu"
#nvcc $file -lcublas -DCUBLASXt -o cublasxt
#nvcc $file -lcublas -DCUBLASN -o cublas
#nvcc $file -lcublas -DCPU -o naivecpu
#nvcc $file -lcublas -DSIMPLECUDA -o naivecuda

