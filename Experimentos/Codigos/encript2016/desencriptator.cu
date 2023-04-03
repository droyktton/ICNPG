/*

	Compílese con
	nvcc -I ./include desencriptator.cu -o desencriptor

*/


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
/*  Descomente el siguiente struct si va a usar thrust
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
*/ 
// Ayuda kernel:
/* descomente el siguiente __global__ void si va a usar un kernel de CUDA
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

// lee el archivo encriptado, procesa los pixels
// genera una imagen desencriptado.pgm en formato pgm lista para visualizar
// El ejecutable recibe un solo argumento que es la imagen a desencriptar
//
int main(int argc, char **argv)
{
	int N=L*L;

	thrust::host_vector<int> hx(N);
	thrust::host_vector<int> hy(N);
	thrust::host_vector<int> hz(N);

	if(argc!=2) {
		printf("Usese así: desencriptor nombre_del_archivo_de_imagen\n");
		exit(1);
	}
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



	// TODO: reordene z de tal forma que los pixels esten en el orden correcto en la imagen
	// Ayuda: reordene z con thrust::sort_by_key() usando como key los indices n=x+L*y 
	// con los x, y ya corregidos. 


	// Acá asumimos que z tiene los pixeles ordenados. Si usó algún otro método
	// no sugerido, asegúrese de que z tenga la info que corresponde.
	// Copiamos los pixels ordenados a hz para imprimir imagen en formato pgm. 
	// https://en.wikipedia.org/wiki/Netpbm_format
	thrust::copy(z.begin(),z.end(),hz.begin());
	std::ofstream desencout("desencriptado.pgm");
	desencout << "P2\n512 512\n255\n";
	for(int n=0;n<N;n++){
		desencout << hz[n] << std::endl; 
	}
	return 0;
}
