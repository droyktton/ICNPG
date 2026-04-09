# Paths & Includes
CUDA_INSTALL_PATH       = 
CUDA_LIB_PATH           = 
INCLUDES += 

#parametros
TAMANIO			= 4096 
PARAMETROS		= -DTAMANIO=$(TAMANIO) -DIMPRIMIR #si imprime asegurese de que TAMANIO sea chico! 

# Flags
LIBS                    = #-L$(CUDA_LIB_PATH)
LDFLAGS                 = -lrt -lm -lcudart -lcufft 
NVCC_FLAGS              = -O2 --use_fast_math   
#OMP_FLAGS        	= -O2 -fopenmp -DTHRUST_HOST_BACKEND=THRUST_HOST_BACKEND_OMP -lgomp -lfftw3 -lm -L/opt/fftw-3.3.3/lib 
FFTW_FLAGS		= -O2 -lfftw3 -lm -lrt -L/share/apps/codigos/fftw3/lib
MULTIFFTW_FLAGS		= -O2 -fopenmp -lgomp -lfftw3_threads -lpthread -lfftw3 -lm -lrt -DOMP -L/share/apps/codigos/fftw3/lib #-L/home/koltona/fftw3/lib
NVCC                    = nvcc 
GCC			= g++
LINKER			= $(NVCC)

# Rules
all: clean simple_cufft_sol

	  
simple_cufft_sol: simple_cufft_solucion.cu
	$(LINKER) $(NVCC_FLAGS) -o simple_cufft_sol simple_cufft_solucion.cu $(PARAMETROS) $(LDFLAGS) $(INCLUDES) $(LIBS)

clean:
	rm -f simple_cufft_sol 
                                                     
