# Paths & Includes
CUSP_PATH		= /share/apps/icnpg/common/
CUSP_PATH2		= /usr/local/cuda-8.0/targets/x86_64-linux/include/
INCLUDES 		+= -I$(CUSP_PATH) -I$(CUSP_PATH2) 

# Flags
NVCC_FLAGS              =  -O2 --use_fast_math 
OMP_FLAGS               = -O2 -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -DOMP

NVCC                    = nvcc
GCC			= g++
LINKER			= $(NVCC)


# Rules
all: clean ejecutableCUDA ejecutableOMP

ejecutableCUDA: sagan.cu 
	$(LINKER) $(NVCC_FLAGS) -o ejecutableCUDA sagan.cu $(LDFLAGS) $(INCLUDES) $(LIBS) $(PARAMETERS)
	  
ejecutableOMP: sagan.cu 
	$(GCC) -x c++ -o ejecutableOMP sagan.cu $(OMP_FLAGS) $(INCLUDES) $(PARAMETERS)
				 
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log ejecutableCUDA ejecutableOMP 
                                                     

runCUDA: ejecutableCUDA
	qsub jobGPU

runOMP: ejecutableOMP
	qsub jobGPU_omp

