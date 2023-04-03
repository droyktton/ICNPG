APPS=convolucion2d_cufft

all: ${APPS}

%: %.cu
	nvcc -O2 -lcufft -DLAPLACIAN -DPRINT -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./conv2d

submit:	all
	qsub jobGPU; watch qstat
