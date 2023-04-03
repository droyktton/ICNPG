#include<fstream>
#include<iostream>
#include<ctime>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/sort.h>
#include<thrust/tuple.h>
#include<thrust/sequence.h>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

// trabajaremos con imagenes de 512x512 == 262144 pixels grises
#define L	512


// TODO: escriba un functor o kernel que corrija las posiciones de cada pixel
// usando la informacion brindada sobre la generacion de los desplazamientos aleatorios
// Ayuda Functor:
/*
struct operacion()
{
	// declaramos estado interno que necesite
	int param1; int param2; ...

	// constructor
	operacion(int p1, int p2, ...):param1(p1),param2(p2),...{};

	// operador "()" 
	__device__
	.... operator()(....)
	{
		RNG philox; 	
	    	RNG::ctr_type c={{}};
    		RNG::key_type k={{}};
    		RNG::ctr_type r;

		k[0]=...; k[1]=0;
		c[0]=...; c[1]=0;

		r=philox(c,k);

		...
	}
};

// Ayuda kernel:
__global__
void kernel(...........)
{
		RNG philox; 	
	    	RNG::ctr_type c={{}};
    		RNG::key_type k={{}};
    		RNG::ctr_type r;

		k[0]=...; k[1]=0;
		c[0]=...; c[1]=0;

		r=philox(c,k);
		.....
}
*/

//////////////////////////////////////////////////////////////////
// Solucion (con thrust):
struct addrandom{
	int fac;
	int seed;
	addrandom(int _fac, int _seed):fac(_fac),seed(_seed){};
	__device__
	thrust::tuple<int,int,int> operator()(int n, thrust::tuple<int,int,int> v)
	{
		RNG philox; 	
	    	RNG::ctr_type c={{}};
    		RNG::key_type k={{}};
    		RNG::ctr_type r;
		k[0]=thrust::get<2>(v); k[1]=0;
		c[0]=seed; c[1]=0;

		r=philox(c,k);

		return thrust::make_tuple(thrust::get<0>(v)+fac*(r[0]%L),thrust::get<1>(v)+fac*(r[1]%L),thrust::get<2>(v));
	}
};
//////////////////////////////////////////////////////////////////

// lee el archivo encriptado, procesa los pixels
// genera una imagen desencriptado.pgm en formato pgm lista para visualizar
int main(int argc, char **argv)
{
	int N=L*L;

	thrust::host_vector<int> hx(N);
	thrust::host_vector<int> hy(N);
	thrust::host_vector<int> hz(N);

	if(argc!=2) exit(1);
	std::ifstream fin(argv[1]);
	for(int n=0;n<N;n++){
		fin >> hx[n] >> hy[n] >> hz[n]; 
	}

	// vectores de device para operar en la GPU
	thrust::device_vector<int> x(hx);
	thrust::device_vector<int> y(hy);
	thrust::device_vector<int> z(hz);
	// si necesita punteros crudos (por ej, en un kernel), aqui los tiene:
	int *rawx=thrust::raw_pointer_cast(&x[0]);
	int *rawy=thrust::raw_pointer_cast(&y[0]);
	int *rawz=thrust::raw_pointer_cast(&z[0]);


	// TODO: substraiga el desplazamiento random de cada pixel en device. 
	// Ayudas: 
	// Primero calcule los parametros que necesita el generador random para pasarle al kernel o functor de thrust de la transfomacion.
	// Luego realice la transformacion de las cordenadas.
	// Solucion:
	// .........
	int seed2=thrust::reduce(z.begin(),z.end());
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  addrandom(-1,seed2)
			);	

	// TODO: reordene z de tal forma que los pixels esten en el orden correcto en la imagen
	// Ayuda: reordene z con thrust::sort_by_key() usando como key los indices n=x+L*y con los x, y ya corregidos. 
	// Solucion: 
	// .........
	using namespace thrust::placeholders;
	thrust::device_vector<int> dindices(N);
	thrust::transform(x.begin(), x.end(),y.begin(),dindices.begin(), _1+_2*L);	
	thrust::sort_by_key(dindices.begin(),dindices.end(),z.begin());

	// copiamos los pixels ordenados a hz para imprimir imagen en formato pgm. 
	// https://en.wikipedia.org/wiki/Netpbm_format
	thrust::copy(z.begin(),z.end(),hz.begin());
	std::ofstream desencout("desencriptado.pgm");
	desencout << "P2\n512 512\n255\n";
	for(int n=0;n<N;n++){
		desencout << hz[n] << std::endl; 
	}
	return 0;
}
