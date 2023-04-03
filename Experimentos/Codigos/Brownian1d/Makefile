NVCC_FLAGS        = -O2 -lcurand
NVCC              = nvcc
LINKER			  = $(NVCC)

# Rules
all: Brown_CUDA

Brown_CUDA: g3ej3curand.cu 
	$(LINKER) $(NVCC_FLAGS) -o Brown_CUDA g3ej3curand.cu 
	  
clean:
	rm -f Brown_CUDA
                                                     

