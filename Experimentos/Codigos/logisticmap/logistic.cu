/*
Itera el mapeo logistico y calcula los exponentes de liapunov en funcion del parametro r.

Entrada: 
las condiciones iniciales y los valores de r que uno quiere calcular, 
el numero de iteraciones, el numero de particulas.

Salida: 
las posiciones finales, y sus exponentes de liapunov en funcion de r.

TODO:
completar el codigo.
al ultimo estaria bueno calcular F(lambda), probabilidad acumulada. 

*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "simple_timer.h"

struct logistic_variable{
	float x;
	float lambda;
};


struct logistic_lyapunov_combined
{
  const unsigned int trun;

  logistic_lyapunov_combined(unsigned int _trun) : trun(_trun) {}

  __host__ __device__
  logistic_variable operator()(float r, float _x) const
  { 
  	float x=_x;
  	float acum=0.0;
  	for(int i=0;i<trun;i++){
  		acum+=logf( fabsf(r-2.0*r*x) );
  		x=r*x*(1.0-x);
  	}
  	logistic_variable res;
  	res.x=x; res.lambda=acum/trun;
  	return res;	
  }

};


// este test calcula los exponentes de lyapunov del mapeo logistico para un rango de r
void test3(int argc, char **argv)
{

	unsigned int N;
	unsigned int trun;
	float rmin;
	float rmax;

	if(argc!=5)
	{	
		std::cout << "uso: " << argv[0] << " N trun rmin rmax" << std::endl;	
		exit(1);
	}	
	else{
		N= atoi(argv[1]);
		trun= atoi(argv[2]);
		rmin= atof(argv[3]);
		rmax= atof(argv[4]);
		std::cout << "#parametros: N=" << N << ", trun=" << trun << ", rmin=" << rmin << ", rmax=" << rmax << std::endl;

		if(rmax>4 || rmin<0){
			std::cout << "ojo: [rmin,rmax] debe estar contenido en [0:4]" << std::endl;
			exit(1);	
		}
	}


    using namespace thrust::placeholders;

    // r distribuido entre [0:4]
	thrust::host_vector<float> h_r(N);
    thrust::sequence(h_r.begin(), h_r.end());
    thrust::transform(h_r.begin(), h_r.end(),h_r.begin(),rmin+_1*(rmax-rmin)/N);

    // un vector para guardar los exponentes de lyapunov
	thrust::host_vector<float> h_x(N);
    std::generate(h_x.begin(), h_x.end(), rand);
    thrust::transform(h_x.begin(), h_x.end(),h_x.begin(),_1/RAND_MAX);

    // un vector para guardar los exponentes de liapunov -> RESULATADO
    logistic_variable zero; zero.x=zero.lambda=0.0;
    thrust::host_vector<logistic_variable> h_resultado(N,zero);

    #ifdef CPUTEST
    gpu_timer crono;
    crono.tic();

    thrust::transform(h_r.begin(),h_r.end(), h_x.begin(), h_resultado.begin(), logistic_lyapunov_combined(trun));

    std::cout << "#ms =" << crono.tac();
    #else
    cpu_timer crono;
    crono.tic();

    thrust::device_vector<float> x(h_x);
    thrust::device_vector<float> r(h_r);
    thrust::device_vector<logistic_variable> resultado(N,zero);
    thrust::transform(r.begin(),r.end(), x.begin(), resultado.begin(), logistic_lyapunov_combined(trun));
    thrust::copy(resultado.begin(), resultado.end(), h_resultado.begin());

    std::cout << "#ms =" << crono.tac();
    #endif

    // imprime los resultados
    for(int i=0;i<N;i++){
    	std::cout << h_r[i] << " " << h_resultado[i].x << " " << h_resultado[i].lambda << std::endl;	
    }
    std::cout << "\n\n";
}

/*
gnuplot
set palette model RGB defined ( 0 'red', 1 'green' ); plot '< ./gpu.out 10000 10000 3.5 4.0' u 1:2:(($3>0)?(0):(1)) palette pt 1
*/

int main(int argc, char **argv){
	test3(argc,argv);
}

