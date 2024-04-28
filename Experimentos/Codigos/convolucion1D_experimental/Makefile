APPS=convolucion

all: ${APPS}

INCLUDES= -I/share/apps/icnpg/common

%: %.cu
	nvcc -O2 -lcufft ${INCLUDES} -o $@ $<

clean:
	rm -f ${APPS} *.dat

run:	
	./convolucion

submit:	
	qsub jobGPU; watch qstat


