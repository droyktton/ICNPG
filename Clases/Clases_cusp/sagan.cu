/*
 * Naive solution of the equation: du(x,y,t)/dt = \lap u(x,y,t) + rho(x,y,t)
 *
 * Compile:  nvcc -I ~/libs/cusplibrary-master/ ecdif.cu
 * Run:		 ./a.out 15 > zzz
 *
 * Pelicula en gnuplot (movie.gnu provisto):
 * gnuplot> set term gif animate; set out 'peli.gif';l=15;i=0;load 'movie.gnu'
 *
 * ABK para icnpg (2015)
*/

#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/elementwise.h>
#include <cusp/multiply.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <cstdlib>
#include<fstream>

// paso de tiempo 
#define Dt	0.1

// paso de euler: u(x,y,t+dt)=u(x,y,t)+dt*(d^2 u/dx^2+ d^2 u/dy^2 + fuente(x,y,t))
struct onestep{
	__device__ __host__ 
	float operator()(thrust::tuple<float,float, float> tup)
	{
		float xold=thrust::get<0>(tup);
		float lapx=-thrust::get<1>(tup);
		float fuente=thrust::get<2>(tup);
		return xold+Dt*(lapx+fuente);
	}
};


// para imprimir en formato matrix de gnuplot
std::ofstream peliout("peli.dat");
template<typename Matrix_type>
void print_matrix(Matrix_type &x, int L){
	for(int i=0;i<L;i++)
	{ 
		for(int j=0;j<L;j++)
		{ 
			peliout << x[j+i*L] << " ";
		}
		peliout << "\n";
	}
	peliout << "\n";
}


int main(int argc, char **argv)
{
	int L=15;
	if(argc>1)
	L=atoi(argv[1]);

	// operador laplaciano en 2D en formato sparse en device
    	cusp::coo_matrix<int, float, cusp::device_memory> A;
    	cusp::gallery::poisson5pt(A, L, L);
	cusp::print(A);

    	// vector fuente en device
    	cusp::array1d<float,cusp::device_memory> b(L*L);

    	// vector campo inicial
    	cusp::array1d<float,cusp::device_memory> x(L*L,0.0f);    

    	// laplaciano aplicado sobre el campo (vector intermedio)
    	cusp::array1d<float,cusp::device_memory> y(L*L);    

    	// posicion de la fuente puntual
    	int i0,j0;

    	using namespace thrust::placeholders;
    	for(int t=0;t<50000;t++)
    	{	
    		// no necesitamos ver todas las configuraciones
    		if(t%20==0) print_matrix(x,L);

    		// cambiamos la posicion de la fuente (se mueve en circulo)
    		i0=int(L*0.5+L*0.3*cos(t*0.001))%L; j0=int(L*0.5+L*0.3*sin(t*0.001))%L;

		// la fuente solo existe en (i0,j0) y vale -1.0 ahi...
    		thrust::fill(b.begin(),b.end(),0.0f);
    		b[int(j0+L*i0)]=-1.0f;

    		// calcula y = A * x, con A operador de Poisson o laplaciano discretizado 
    		cusp::multiply(A, x, y);

    		// compute x <- x + (A*x + b)dt
		#if __cplusplus!=201103L

    		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(), b.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end(), b.end())),
			x.begin(), 
			onestep()
    		);

		#else 

		// esta forma hace lo mismo pero es mas simple y clara (compilar con nvcc --std=c++11)
		auto begin=thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(), b.begin()));
		auto end= begin + L*L;
    		thrust::transform(begin, end, x.begin(), onestep());

		#endif
	}	

    return 0;
}
