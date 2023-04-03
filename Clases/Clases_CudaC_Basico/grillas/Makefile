APPS=grillas

NTHX?=5
NTHY?=1
NTHZ?=1
NBLX?=1
NBLY?=1
NBLZ?=1

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./grillas $(NTHX) $(NTHY) $(NTHZ) $(NBLX) $(NBLY) $(NBLZ)

submit:	all
	qsub jobGPU $(NTHX) $(NTHY) $(NTHZ) $(NBLX) $(NBLY) $(NBLZ)

submitwatch:	all
	qsub jobGPU $(NTHX) $(NTHY) $(NTHZ) $(NBLX) $(NBLY) $(NBLZ); watch qstat
