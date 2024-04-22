APPS=convolucion

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./convolucion

submit:	all
	qsub jobGPU; watch qstat
