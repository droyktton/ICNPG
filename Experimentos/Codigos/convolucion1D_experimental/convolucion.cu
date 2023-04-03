#include <cstdio>
#include <cstdlib>
#include <cassert>
//#include "curso.h"
#include "gpu_timer.h"
#include "cpu_timer.h"
#include<thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

// CUFFT include http://docs.nvidia.com/cuda/cufft/index.html
#include <cufft.h>
#include "cutil.h"	// CUDA_SAFE_CALL, CUT_CHECK_ERROR


/* Kernel Execution Parameters Parameters */
const int BLOCK_SIZE = 512;
const int BLOCK_SIZEX = 4;
const int BLOCK_SIZEY = 256;

/* Size of the input data */
#ifndef SIZE
//const int N = 4194304;  
const int N = 8388608;  
//const int N = 65536;  
#else
const int N = SIZE;
#endif

/* Size of the filter */
#ifndef FILTERSIZE
//const int Nh = 128;
const int Nh = 4;
#else
const int Nh = FILTERSIZE;
#endif

/* Size of the window */
//const int M = 128;
#ifndef WINDOWSIZE
const int M = 4096;
#else
const int M = WINDOWSIZE;
#endif

/* Floating point type */
typedef float FLOAT;

/* Function to setup the filter */
static void SetupFilter(FLOAT* filter, size_t size, void* user) {
	for(int i = 0 ; i < size ; i++)
		filter[i] = 1.0/size;
}


/* convolucion en la cpu: requiere dos loops */
void conv_cpu(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{
	FLOAT temp;
	for(int j=0;j<N;j++){
		temp=0.0;
		for(int i=0;i<Nh;i++){
	  		temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}

// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// todo en memoria global
// lanzamiento: la grilla se puede elegir independiente de N
__global__ void conv_one_thread_per_output_element_all_global
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += filter[i]*input[i+j];
	  	}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}

// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// filtro en memoria constante, el resto en global
// lanzamiento: la grilla se puede elegir independiente de N
__constant__ FLOAT d_filtro_constant[Nh];

__global__ void conv_one_thread_per_output_element_filter_in_constant
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += d_filtro_constant[i]*input[i+j]; // cuidado: solo 64K de constant memory
	  	}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}


struct conv_one_thread_per_output_element_filter_in_constant_functor
{
	FLOAT* senial;
	conv_one_thread_per_output_element_filter_in_constant_functor
	(FLOAT* input)
	{
		senial=input;
	};
	__device__ 
	FLOAT operator()(const int j)
	{
		FLOAT temp;
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += d_filtro_constant[i]*senial[i+j];
	  	}	  
		return temp;
	}
};

#include <thrust/inner_product.h>
struct conv_one_thread_per_output_element_filter_in_constant_functor_thrust18
{
	FLOAT* senial;
	conv_one_thread_per_output_element_filter_in_constant_functor_thrust18
	(FLOAT* input)
	{
		senial=input;
	};
	__device__ 
	FLOAT operator()(const int j)
	{
		FLOAT temp = thrust::inner_product(thrust::seq, d_filtro_constant, d_filtro_constant+Nh, senial+j,float(0.0)); 	  
		return temp;
	}
};




// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// filtro en memoria shared, datos en global
// lanzamiento: la grilla se puede elegir independiente de N y Nh
__global__ void conv_one_thread_per_output_element_filter_in_shared
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  __shared__ FLOAT sh_filter[Nh];
	  int j = blockIdx.x * blockDim.x + threadIdx.x;
	  int tx = threadIdx.x;	

	  // Como blockDim.x puede ser menor a Nh, me aseguro 
	  // que los blockDim.x threads carguen todo el filtro 
	  // dandoles como tarea cargar mas de un elemento, si fuera necesario
	  while(tx<Nh){
		sh_filter[tx]=filter[tx];
		tx+=blockDim.x;
	  }
	  __syncthreads();

	  FLOAT temp; 
	  while(j<N)
	  {
	  	temp=0.0; 
	  	for(int i=0;i<Nh;i++){
	  		temp += sh_filter[i]*input[i+j];
	  	}	  
	  	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}



// convolucion usando indexado unidimensional de threads/blocks
// un bloque calcula cada elemento del output (que se queda en global)
// filtro en memoria shared, datos segmentados en ventanas cargadas en shared
__global__ void conv_one_thread_per_output_element_all_in_shared
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int k = blockIdx.x;
	  int tx = threadIdx.x;	

	  __shared__ FLOAT sh_filter[Nh];  // sh_filter[i] = filter[i]
	  __shared__ FLOAT sh_input[M+Nh-1]; // sh_input[i]  <= input[k*M+i] 

	  while(k*M<N)// caso gridDim.x*M < N contemplado 
	  { 
		  // cargo filtro
		  // caso blockDim.x < Nh contemplado	
		  tx = threadIdx.x;	
		  while(tx<Nh){	
			sh_filter[tx]=filter[tx]; 
			tx+=blockDim.x;
		  }

		  // carga ventana con padding 	
		  // caso blockDim.x < Nh+M contemplado	
		  tx = threadIdx.x;	
		  while(tx<M+Nh){ 
			sh_input[tx]=input[M*k+tx];
			tx+=blockDim.x;		    	
		  }	
		  __syncthreads();	 	

		  // aqui cada thread del bloque "k" calcula uno de estos: output[k*M],...,output[(k+1)*M-1]
		  // caso blockDim.x < M contemplado	
		  tx = threadIdx.x;		  
		  FLOAT temp;	  
		  while(tx<M){
			temp=0.0;
			for(int i=0;i<Nh;i++){
				temp += sh_filter[i]*sh_input[i+tx];
			}	  	  	  		
			output[k*M+tx] = temp; 
			tx+=blockDim.x;		    	
		  }
		  k+=gridDim.x;
		  __syncthreads();// no puedo cargar mas datos si todo el output no esta listo...	 	
	  }
}



// convolucion usando indexado unidimensional de threads/blocks
// un bloque calcula cada elemento del output (que se queda en global)
// filtro en memoria constante, datos segmentados en ventanas cargadas en shared
__global__ void conv_one_thread_per_output_element_input_in_shared_filter_in_constant
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int k = blockIdx.x;
	  int tx = threadIdx.x;	

	  __shared__ FLOAT sh_input[M+Nh-1]; // sh_input[i]  <= input[k*M+i] 

	  while(k*M<N)// caso gridDim.x*M < N contemplado 
	  { 
		  // carga ventana con padding 	
		  // caso blockDim.x < Nh+M contemplado	
		  tx = threadIdx.x;	
		  while(tx<M+Nh){ 
			sh_input[tx]=input[M*k+tx];
			tx+=blockDim.x;		    	
		  }	
		  __syncthreads();	 	

		  // aqui cada thread del bloque "k" calcula uno de estos: output[k*M],...,output[(k+1)*M-1]
		  // caso blockDim.x < M contemplado	
		  tx = threadIdx.x;		  
		  FLOAT temp;
	  
		  while(tx<M){
			temp=0.0;
			for(int i=0;i<Nh;i++){
				temp += d_filtro_constant[i]*sh_input[i+tx];
			}	  	  	  		
			output[k*M+tx] = temp; 
			tx+=blockDim.x;		    	
		  }
		  k+=gridDim.x;
		  __syncthreads();	 	
	  }
}


////////////////////////////////////////////////////////////////
// LOS TRES KERNELS SIGUIENTES SON MALISIMOS: COMPRUEBELO Y PIENSE PORQUE...

// Kernel de tres lineas!!!!, pero es malisimo...
// Convolucion con indexado bidimensional de threads/blocks
// Matriz[j, i] = filter[i] input[i + j] :
// output[j] = Suma_{i} Matriz[j, i]  
// plan: que cada thread calcule un elemento de Matriz
__global__ void conv_atomic
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	

	  // filter[i]: garantizar (0 <= i < Nh)
	  int i = blockIdx.x * blockDim.x + threadIdx.x;

	  // output[j]: (garantizar 0 <= j < N)
	  int j = blockIdx.y * blockDim.y + threadIdx.y;

	  // solucion usando atomicAdd:

	  // esta supone que i y j barren bien el filtro y el input
	  /*if(i<Nh && j < N){
	  	atomicAdd( output+j, (filter[i]*input[i+j]));	
	  }*/	

	  // esta no supone que i y j barren bien el filtro y el input 
	  // -> serializa para que alcance el numero de threads para hacer todo el trabajo
	  while(i<Nh)
	  {
	  	while(j<N)
	  	{
	  		atomicAdd( output+j, (filter[i]*input[i+j]));
			j+=gridDim.y*blockDim.y;
		}
		i+=gridDim.x*blockDim.x;
	  };	
}


// Kernel de tres lineas!!!!, pero es malisimo...
// Convolucion con indexado bidimensional de threads/blocks
// Matriz[j, i] = filter[i] input[i + j] :
// output[j] = Suma_{i} Matriz[j, i]  
// plan: que cada thread calcule un elemento de Matriz
__global__ void conv_atomic_filter_in_constant
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	

	  // filter[i]: garantizar (0 <= i < Nh)
	  int i = blockIdx.x * blockDim.x + threadIdx.x;

	  // output[j]: (garantizar 0 <= j < N)
	  int j = blockIdx.y * blockDim.y + threadIdx.y;

	  // solucion usando atomicAdd:
	  // esta supone que i y j barren bien el filtro y el input
	  /*if(i<Nh && j < N){
	  	atomicAdd( output+j, (filter[i]*input[i+j]));	
	  }*/	

	  // esta no supone que i y j barren bien el filtro y el input 
	  // -> serializa para que alcance el numero de threads para hacer todo el trabajo
	  while(i<Nh)
	  {
	  	while(j<N)
	  	{
	  		atomicAdd( output+j, (d_filtro_constant[i]*input[i+j]));
			j+=gridDim.y*blockDim.y;
		}
		i+=gridDim.x*blockDim.x;
	  };	
}

// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// todo en memoria global, pero el filtro se carga en registros...
// lanzamiento: la grilla se puede elegir independiente de N
__global__ void conv_one_thread_per_output_element_input_global_filter_in_register
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;
	 
          FLOAT r_filter[Nh];
	  for(int i=0;i<Nh;i++) r_filter[i]=filter[i];

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += r_filter[i]*input[i+j];
	  	}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}

__global__ void multiply_complex(cufftComplex *d_f_filter,cufftComplex *d_f_input,cufftComplex *d_f_output)
{
	  int j = blockIdx.x * blockDim.x + threadIdx.x;
	  while(j<N)
	  {
		// z = x+iy -> z1.z2* = (x1 + i y1)*(x2 - iy2) = x1.x2 + y1.y2 + i(y1.x2 - y2.x1);
		d_f_output[j].x = (d_f_filter[j].x*d_f_input[j].x + d_f_filter[j].y*d_f_input[j].y)/(N);
		d_f_output[j].y = (d_f_filter[j].x*d_f_input[j].y - d_f_filter[j].y*d_f_input[j].x)/(N);

		  // z = x+iy -> z1.z2  = (x1 + i y1)*(x2 + iy2) = x1.x2 - y1.y2 + i(y1.x2 + y2.x1);
		/*d_f_output[j].x = (d_f_filter[j].x*d_f_input[j].x - d_f_filter[j].y*d_f_input[j].y)/N;
		d_f_output[j].y = (d_f_filter[j].x*d_f_input[j].y + d_f_filter[j].y*d_f_input[j].x)/N;*/

		j+=gridDim.x*blockDim.x;
	  }
}

////////////////////////////////////////////////////////////////
// Para usar con CUSP
#include <cusp/linear_operator.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/array1d.h>
class stencil : public cusp::linear_operator<float,cusp::device_memory>
{
	public:
    typedef cusp::linear_operator<float,cusp::device_memory> super;

    int NN;

    // constructor
    stencil(int NN)
        : super(NN,NN), NN(NN) {}

    // linear operator y = A*x
    template <typename VectorType1,
             typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        const float * x_ptr = thrust::raw_pointer_cast(&x[0]);
        float * y_ptr = thrust::raw_pointer_cast(&y[0]);

		conv_one_thread_per_output_element_input_in_shared_filter_in_constant
		<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(x_ptr, y_ptr, x_ptr);

		//conv_one_thread_per_output_element_filter_in_constant
		//<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(x_ptr, y_ptr, x_ptr);
    }
};
////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
	cudaDeviceProp deviceProp;
    	int dev; cudaGetDevice(&dev);
    	//int dev=1; cudaSetDevice(dev);
    	cudaGetDeviceProperties(&deviceProp, dev);
    	printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);


	int caso;
	if(argc==2)
	{
		caso=atoi(argv[1]);
		printf("caso %d\n",caso);
	}
	else{
		printf("uso %s #caso\n",argv[0]);

		printf("casos: \n");
		printf("1- filtro e input en memoria global\n");
		printf("2- filtro en memoria constante, input en memoria global\n");
		printf("3- filtro en memoria shared, input en memoria global\n");
		printf("4- filtro en memoria shared, input en memoria shared\n");
		printf("5- filtro en memoria constante, input en memoria shared\n");
		printf("6- filtro en registro... input en memoria global \n");
		printf("7- Convolucion con transformadas de Fourier... \n");
		printf("8- Convolucion con thrust: input en global, filtro en constant \n");
		printf("9- Convolucion con cusp (no implementado): input en global, filtro en constant \n");
		printf("10- filtro e input en memoria global, 1 thread por operacion * con atomics\n");
		printf("11- filtro en memoria constante, input en memoria global, 1 thread por operacion * con atomics \n");

		exit(1);
	}

	/* Print general information */
	printf("[@] Size of the array: %d\n", N);
	printf("[@] Size of the filter array: %d\n", Nh);
	printf("[@] Size of the windows: %d\n", M);

	/* Allocate memory on host */
	FLOAT* h_input = (FLOAT *) malloc((N+Nh) * sizeof(FLOAT));  /* Input data */
	FLOAT* h_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* Output data */
	FLOAT* check_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* CPU Output data */
	/* Allocate memory for filter */
	FLOAT* h_filter = (FLOAT*) malloc(Nh * sizeof(FLOAT));


	/* Setup the filter */
	SetupFilter(h_filter, Nh, 0);

	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) h_input[i] = (FLOAT)(rand() % 100); 
	for(int i = N ; i < N+Nh ; i++) h_input[i] = h_input[i-N];


	/* Allocate memory on device */
	FLOAT *d_input, *d_output, *d_filter, *d_extended_filter;
	cudaMalloc((void**)&d_input, (N+Nh) * sizeof(FLOAT));
	cudaMalloc((void**)&d_output, N * sizeof(FLOAT));
	cudaMalloc((void**)&d_filter, Nh * sizeof(FLOAT));
	cudaMalloc((void**)&d_extended_filter, N * sizeof(FLOAT));

	cufftComplex *d_f_input, *d_f_output, *d_f_filter;
	cudaMalloc((void**)&d_f_input,  (N/2+1) * sizeof(cufftComplex));
	cudaMalloc((void**)&d_f_output, (N/2+1) * sizeof(cufftComplex));
	cudaMalloc((void**)&d_f_filter, (N/2+1) * sizeof(cufftComplex));
	cudaMalloc((void**)&d_extended_filter, N * sizeof(FLOAT));

	cufftHandle plan_r2c;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_r2c,N,CUFFT_R2C,1));
	cufftHandle plan_c2r;
	CUFFT_SAFE_CALL(cufftPlan1d(&plan_c2r,N,CUFFT_C2R,1));


	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
	cudaMemset(d_extended_filter,0,N * sizeof(FLOAT));

	/* Copy input array to device */
	cudaMemcpy(d_input, h_input, (N+Nh) * sizeof(FLOAT), cudaMemcpyHostToDevice);
	checkCUDAError("Memcpy input array : ");

	/* Copy the filter to the GPU */
	cudaMemcpy(d_filter, h_filter, Nh * sizeof(FLOAT), cudaMemcpyHostToDevice);
	checkCUDAError("Memcpy filter array : ");
	cudaMemcpy(d_extended_filter, d_filter, Nh * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
	checkCUDAError("Memcpy filter array : ");

	/* Copy the filter to the GPU in constant memory */
	cudaMemcpyToSymbol(d_filtro_constant,h_filter,sizeof(FLOAT)*Nh);
	checkCUDAError("Memcpytosymbol filter array : ");

    stencil STENCIL(N);
	// use array1d_view to represent the linear array data
    typedef typename cusp::array1d_view< thrust::device_ptr<float> > DeviceArray1dView;
	thrust::device_ptr<float> dev_ptr_input = thrust::device_pointer_cast(d_input);
	thrust::device_ptr<float> dev_ptr_output = thrust::device_pointer_cast(d_output);
	DeviceArray1dView cusp_input(dev_ptr_input,dev_ptr_input+N+Nh);
	DeviceArray1dView cusp_output(dev_ptr_output,dev_ptr_output+N);

	/* Sanity check */
	assert(Nh <= M);
	assert(M <= N);

	/* check in the CPU */
	cpu_timer cronocpu; cronocpu.tic();
	conv_cpu(h_input, check_output, h_filter);
	cronocpu.tac();

	assert(BLOCK_SIZEX*BLOCK_SIZEY<1025);// 1024 threads per block
	dim3 block_size(BLOCK_SIZEX,BLOCK_SIZEY);
  	dim3 grid_size((Nh + BLOCK_SIZEX -1) / BLOCK_SIZEX,(N + BLOCK_SIZEY -1)  / BLOCK_SIZEY);

	/* distintos kernels */
	gpu_timer crono;
	crono.tic();
	switch(caso)
	{
		case 1:
		printf("filtro e input en memoria global\n");
			conv_one_thread_per_output_element_all_global<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv_one_thread_per_output_element_all_global) Kernel invocation: ");
		break;

		case 2:
		printf("filtro en memoria constante, input en memoria global\n");
	        conv_one_thread_per_output_element_filter_in_constant
		<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv_one_thread_per_output_element_filter_in_constant) Kernel invocation: ");
		break;

		case 3:
		printf("filtro en memoria shared, input en memoria global\n");
	        conv_one_thread_per_output_element_filter_in_shared
		<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv_one_thread_per_output_element_filter_in_shared) Kernel invocation: ");
		break;

		case 4:
		printf("filtro en memoria shared, input en memoria shared\n");
	        conv_one_thread_per_output_element_all_in_shared<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv_one_thread_per_output_element_all_in_shared) Kernel invocation: ");
		break;

		case 5:
		printf("filtro en memoria constante, input en memoria shared\n");
	        conv_one_thread_per_output_element_input_in_shared_filter_in_constant<<<N/BLOCK_SIZE,BLOCK_SIZE>>>
		(d_input, d_output,d_filter);
		checkCUDAError("(conv_one_thread_per_output_element_input_in_shared_filter_in_constant) Kernel invocation: ");
		break;

		case 6:
		printf("filtro en registro... input en memoria global \n");
		conv_one_thread_per_output_element_input_global_filter_in_register<<<N/BLOCK_SIZE,BLOCK_SIZE>>>
		(d_input, d_output, d_filter);
		checkCUDAError("(filter in registers) Kernel invocation: ");
		break;

		case 7:
		printf("Convolucion con transformadas de Fourier... \n");
		CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, d_extended_filter, d_f_filter));
		CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, d_input, d_f_input));
		// kernel de multiplicacion...
		multiply_complex<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_f_filter,d_f_input,d_f_output);
		CUFFT_SAFE_CALL(cufftExecC2R(plan_c2r, d_f_output, d_output));
		checkCUDAError("(fourier) Kernels invocation: ");
		break;

		case 8:
		// hacerlo con un transform de thrust...
		printf("Convolucion con thrust: input en global, filtro en constant \n");
		thrust::transform(
			thrust::cuda::par,
			thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
			d_output,
			conv_one_thread_per_output_element_filter_in_constant_functor(d_input)
		); 
		break;

		case 9:
		// hacerlo con un cusp...
		printf("Convolucion con cusp: input en global, filtro en constant \n");
	    // create a matrix-free linear operator
	    cusp::multiply(STENCIL, cusp_input, cusp_output);
		break;

		case 10:
		printf("filtro e input en memoria global, 1 thread por operacion * \n");
		//printf("gx=%d gy=%d bx=%d by=%d\n", grid_size.x, grid_size.y, block_size.x, block_size.y);
        	conv_atomic<<<grid_size,block_size>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv atomic) Kernel invocation: ");
		break;

		case 11:
		printf("filtro en memoria constante, input en memoria global, 1 thread por operacion * \n");
		//printf("gx=%d gy=%d bx=%d by=%d\n", grid_size.x, grid_size.y, block_size.x, block_size.y);
        	conv_atomic_filter_in_constant<<<grid_size,block_size>>>(d_input, d_output, d_filter);
		checkCUDAError("(conv atomic filter in constant) Kernel invocation: ");
		break;

		default:
		printf("debe seleccionar un caso del 1 al 11\n"); exit(1);
		break;

	}
	cudaThreadSynchronize();
	crono.tac();
	printf("[Nh/N/ms_cpu/ms_gpu]= %d %d %lf %lf\n", Nh, N, cronocpu.ms_elapsed, crono.ms_elapsed);

	/* Copy output array to host */
	cudaMemcpy(h_output, d_output, N * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy output array : ");

	/* comparacion */
	FLOAT error, maxerror;
	for(int j=0;j<N;j++){
		// descomentar para imprimir output
		//printf("%d %f %f %f\n",j, h_input[j], h_output[j], check_output[j]);
		error = fabs(h_output[j]-check_output[j]); //'*100/fabs(check_output[j]);
		if(maxerror<error) maxerror=error;
	}
	printf("error maximo, emax = %lf \n", maxerror);

	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);

	/* Free memory on device */
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);
	cudaFree(d_f_filter);
	cudaFree(d_f_output);
	cudaFree(d_extended_filter);
}

