#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include "simple_timer.h"

// omp backend
//  g++ -x c++ -O2 -o conv convolucion1d_solo_thrust.cu -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I /usr/local/cuda-5.5/include/

// paralelismo dinamico con cuda
// nvcc -arch=sm_35 -rdc=true convolucion1d_solo_thrust.cu -lcudadevrt -o prog -DTHRUST18par
struct conv
{
	float* senial;
	float* filtro;
	int N;
	int Nh;
	conv
	(float* _senial, float *_filtro, int _N, int _Nh)
	{
		senial=_senial;
		filtro=_filtro;
		N=_N;
		Nh=_Nh;
	};

#ifdef THRUST18seq
	__device__ __host__
	float operator()(const int j)
	{
		return thrust::inner_product(thrust::seq, filtro, filtro+Nh, senial+j,float(0.0))/Nh; 	  
	}
#elif defined(THRUST18par)
	#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
	__device__ __host__
	float operator()(const int j)
	{
		return thrust::inner_product(thrust::omp::par, filtro, filtro+Nh, senial+j,float(0.0))/Nh; 	  
	}
	#else // requiere paralelismo dinamico
	__device__ __host__
	float operator()(const int j)
	{
		return thrust::inner_product(thrust::cuda::par, filtro, filtro+Nh, senial+j,float(0.0))/Nh; 	  
	}
	#endif
#else
	__device__ __host__
	float operator()(const int j)
	{
		float temp;
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += filtro[i]*senial[i+j];
	  	}	  
		return temp/Nh;
	}
#endif
};

float ran01()
{
	return float(rand())/RAND_MAX;
}

int main(int argc, char **argv)
{
	int N=(argc>1)?(atoi(argv[1])):(1024);
	int Nh=(argc>2)?(atoi(argv[2])):(512);

	#ifdef THRUST18seq
	gpu_timer crono;
	std::cout << "parallel transform, seq thrust inner product"<< std::endl;
	#elif defined(THRUST18par)
		#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
		omp_timer crono;
		std::cout << "parallel transform, parallel thrust inner product omp (paralelismo dinamico)"<< std::endl;
		#else // cuda parallel
		gpu_timer crono;
		std::cout << "parallel transform, paralell thrust inner product cuda (paralelismo dinamico)"<< std::endl;
		#endif
	#else
	gpu_timer crono;
	std::cout << "parallel transform, seq for loop"<< std::endl;		
	#endif

	thrust::host_vector<float> h_x(N+Nh);
	thrust::host_vector<float> h_f(Nh,float(1.0));

	thrust::generate(h_x.begin(),h_x.end(),ran01);

	thrust::device_vector<float> d_x(h_x);
	thrust::device_vector<float> d_f(h_f);
	thrust::device_vector<float> d_y(N);

	float *d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);	
	float *d_f_ptr = thrust::raw_pointer_cast(&d_f[0]);	
	float *d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);	

	crono.tic();
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),d_y.begin(),conv(d_x_ptr,d_f_ptr,N,Nh));
	std::cout << "ms=" << crono.tac() << std::endl;

	thrust::host_vector<float> h_y(d_y);

	std::ofstream fout("output.dat");
	for(int i=0;i<N;i++){
		fout << h_y[i] << " " << h_x[i] << std::endl;
	}
}
