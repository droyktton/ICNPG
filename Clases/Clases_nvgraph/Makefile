# Paths & Includes
INCLUDES 		+= -I/state/partition1/apps/spack/opt/spack/linux-centos7-x86_64/gcc-8.2.0/cuda-8.0.61-775ideynz7rzyz5mhv4rpdptuu4px4f5/samples/common/inc/

# Flags
NVCC_FLAGS              =  -O2 --use_fast_math -lnvgraph

NVCC                    = nvcc
GCC			= g++
LINKER			= $(NVCC)


# Rules
all: clean ejecutableCUDA

ejecutableCUDA: nvgraph_SSSP.cu
	$(LINKER) $(NVCC_FLAGS) -o ejecutableCUDA nvgraph_SSSP.cu $(LDFLAGS) $(INCLUDES) $(LIBS) $(PARAMETERS)
	  
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log ejecutableCUDA
                                                     
runCUDA: ejecutableCUDA
	qsub jobGPU


