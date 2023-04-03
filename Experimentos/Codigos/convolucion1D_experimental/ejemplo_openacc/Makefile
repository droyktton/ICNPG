SHELL := /bin/bash
CC       = pgc++
CCFLAGS  = 
ACCFLAGS = -acc -ta=nvidia -Minfo=accel 
OMPFLAGS = -fast -mp -Minfo

GCC       = g++ -O2
GNUOMPFLAGS = -fopenmp -lgomp

BIN =  convolucion1d_omp convolucion1d_acc convolucion1d_gnuomp

all: $(BIN)

convolucion1d_acc: convolucion_openacc.cpp
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $@ $<

convolucion1d_omp: convolucion_openacc.cpp
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $<

convolucion1d_gnuomp: convolucion_openacc.cpp
	$(GCC) $(CCFLAGS) $(GNUOMPFLAGS) -o $@ $<

clean:
	$(RM) $(BIN)

