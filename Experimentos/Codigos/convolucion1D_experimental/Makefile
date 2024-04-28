APPS=convolucion

all: ${APPS}

FILTERW=32

INCLUDES= -I/share/apps/icnpg/common

%: %.cu
	nvcc -O2 -lcufft ${INCLUDES} -DFILTERSIZE=${FILTERW} -o $@ $<

clean:
	rm -f ${APPS} *.dat

run:	
	./convolucion

submit:	
	qsub jobGPU; watch qstat


