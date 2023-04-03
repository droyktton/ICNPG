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

#define L	512

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

// lee pgm, desplaza los pixel, e imprime pixels desplazados x,y,z
int main(int argc, char **argv)
{
	std::ifstream finpgm(argv[1]);

	std::string nombre;
	std::string doscincocinco("255");
	do{
		std::getline(finpgm,nombre);
		//std::cout << nombre << " ";
	}while(nombre.compare(doscincocinco)!=0);
	//std::cout << std::endl;

	int N=L*L;
	thrust::host_vector<int> hx(N);
	thrust::host_vector<int> hy(N);
	thrust::host_vector<int> hz(N);

	for(int j=0;j<L;j++)
	{
		for(int i=0;i<L;i++)
		{
			int value;
			finpgm >> value;
			//std::cout << value << std::endl;
			int n=i+j*L;
			hx[n]=i;
			hy[n]=j;
			hz[n]=value;
			//std::cout << hx[n] << " " << hy[n] << " " << hz[n] << std::endl; 
		}		
	}
	std::cout << "\n\n";

	thrust::device_vector<int> x(hx);
	thrust::device_vector<int> y(hy);
	thrust::device_vector<int> z(hz);

	int seed=thrust::reduce(z.begin(),z.end());
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  addrandom(1, seed)
			);

	thrust::copy(x.begin(),x.end(),hx.begin());
	thrust::copy(y.begin(),y.end(),hy.begin());

	char nom[50];
	sprintf(nom,"%s_encripted.dat",argv[1]);
	std::ofstream fout(nom);
	
	std::srand ( unsigned ( std::time(0) ) );
	thrust::host_vector<int> indices(N);
	thrust::sequence(indices.begin(), indices.end());
	std::random_shuffle ( indices.begin(), indices.end() );

	// los pixels conservan su brillo, pero cambiaron posiciones...
	for(int k=0;k<N;k++){
		int n=indices[k];
		fout << hx[n] << " " << hy[n] << " " << hz[n] << std::endl; 
	}

	//////////////////////////////////////////////////////////////////
	// esta seria la solucion del problema
	std::ifstream fin(nom);
	for(int j=0;j<L;j++){
	for(int i=0;i<L;i++){
		int n=i+j*L;
		fin >> hx[n] >> hy[n] >> hz[n]; 
	}}

	thrust::copy(hx.begin(),hx.end(),x.begin());
	thrust::copy(hy.begin(),hy.end(),y.begin());
	thrust::copy(hz.begin(),hz.end(),z.begin());

	// substraigo el desplazamiento
	int seed2=thrust::reduce(z.begin(),z.end());
	thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin())),
			  addrandom(-1,seed2)
			);	

	// reordeno los pixels 
	using namespace thrust::placeholders;
	thrust::device_vector<int> dindices(N);
	thrust::transform(x.begin(), x.end(),y.begin(),dindices.begin(), _1+_2*L);	
	thrust::sort_by_key(dindices.begin(),dindices.end(),z.begin());

	//thrust::copy(x.begin(),x.end(),hx.begin());
	//thrust::copy(y.begin(),y.end(),hy.begin());
	thrust::copy(z.begin(),z.end(),hz.begin());

	std::ofstream desencout("desencriptado.pgm");
	desencout << "P2\n512 512\n255\n";
	for(int n=0;n<N;n++){
		desencout << hz[n] << std::endl; 
	}
	return 0;
}
