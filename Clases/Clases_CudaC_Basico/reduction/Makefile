APPS=reduccion

N ?= 65536
CASO ?= 1

all: ${APPS}

%: %.cu
	nvcc -O2 -o $@ $<

clean:
	rm -f ${APPS} *.dat

run:	all
	./reduccion $(N) $(CASO)

submit:	
	qsub jobGPU; watch qstat

