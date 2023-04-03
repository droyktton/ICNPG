#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/sequence.h>
#include<thrust/transform.h>
#include<thrust/reduce.h>
#include<thrust/transform_reduce.h>
#include<thrust/tuple.h>
#include<thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>
#include <iostream>

struct trapeciator
{
	__host__ __device__
	float operator()(float x,float xp1)
	{
		return (xp1-x)*(sin(x)+sin(xp1))*0.5;		
	}	
};

struct trapeciator_tuple
{
	__host__ __device__
	float operator()(thrust::tuple<float, float> X)
	{
		float x=thrust::get<0>(X);
		float xp1=thrust::get<1>(X);
		return (xp1-x)*(sin(x)+sin(xp1))*0.5;		
	}	
};

struct trapeciator_tipo_kernel //:public thrust::unary_function<int,float>
{
	float *P;
	trapeciator_tipo_kernel(float *_P):P(_P){};

	__host__ __device__
	float operator()(int i)
	{
		float x=P[i];
		float xp1=P[i+1];
		return (xp1-x)*(sin(x)+sin(xp1))*0.5;		
	}	
};


int main(int argc, char **argv)
{
	const int N=(argc>1)?(atoi(argv[1])):(8192);
	std::cout << "N=" << N << std::endl;

	thrust::device_vector<float> x(N);

	// x={0,1/N, 2/N, ...., (N-1)/N}
	thrust::sequence(x.begin(),x.end(),float(0.0),float(1.0/(N-1)));
	
	std::cout << "la integral de sin(x) entre 0 y 1 deberia ser 1-cos(1)=0.45969769413186, no?... veamos:" << std::endl;


	// metodo 1: usando vector intermedio y
	thrust::device_vector<float> y(N);
	thrust::transform(x.begin(),x.end()-1, x.begin()+1, y.begin(), trapeciator());
	std::cout << "aqui me dio: " << thrust::reduce(y.begin(),y.end()) << std::endl;



	// metodo 2: 
	typedef thrust::device_vector<float>::iterator FloatIterator;
	thrust::tuple<FloatIterator,FloatIterator> first = thrust::make_tuple(x.begin(),x.begin()+1);
	thrust::tuple<FloatIterator,FloatIterator> last = thrust::make_tuple(x.end()-1,x.end());

	std::cout << "aqui me dio: " << 
	thrust::transform_reduce(
		thrust::make_zip_iterator(first),
		thrust::make_zip_iterator(last),
		trapeciator_tuple(),
		float(0.0), thrust::plus<float>()
	) << std::endl;




	// metodo 3: 

	float *ptr=thrust::raw_pointer_cast(&x[0]);
	trapeciator_tipo_kernel operacion(ptr);

	std::cout << "aqui me dio: " << 
	thrust::transform_reduce(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(N-1),
		operacion,
		float(0.0), thrust::plus<float>()
	) << std::endl;


}
// si quiere correr en la placa, reemplace host -> device
// compruebe que el metodo 2, que usa kernel fusion es mas rapido para N grande. Use timers. 
// cual es mejor, el 2 o el 3?
// g++ -x c++ -O2 trapecios.cu -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -lgomp -I /usr/local/cuda-5.5/include/
