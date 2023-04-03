#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/copy.h>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#define PARES	0
#define IMPARES	1
#define BLANCO	0
#define NEGRO	1

__device__
float uniform(int n, int seed, int t)
{
		// keys and counters 
		RNG philox; 	
    		RNG::ctr_type c={{}};
    		RNG::key_type k={{}};
    		RNG::ctr_type r;
    		// Garantiza una secuencia random "unica" para cada thread	
    		k[0]=n; 
    		c[1]=seed;
    		c[0]=t;
		r = philox(c, k); 
     		return (u01_closed_closed_32_53(r[0]));
}


struct metropolitor
{
	int *M;
	float temp;
	int L;
	int shift;
	int N;
	int t;
	int seed;

	metropolitor(int *_M, float _temp, int _L, int _shift, int _t, int _seed):
	M(_M),temp(_temp),L(_L),shift(_shift),t(_t), seed(_seed)
	{N=L*L;};	

	__device__
	void operator()(int i)
	{
		
		int n = 2*i + (int((2.0*i)/L)%2)*(1-2*shift)+shift;
		//printf("%d ",n);		
		
		// con condiciones de contorno periodicas en y pero helicas con paso 1 en X...
		int ny = int(n/L);
		int nx = (n%L);	

		int izq=(nx-1+L)%L+ny*L;
		int der=(nx+1)%L+ny*L;
		int arr=nx+((ny-1+L)%L)*L;
		int aba=nx+((ny+1)%L)*L;

		// la magnetizacion total de los vecinos
		int vecinos=M[izq]+M[der]+M[arr]+M[aba];
		
		// contribucion de nuestro spin sin flipear a la energia  
		int ene0=-M[n]*vecinos;	

		// contribucion a la energia de nuestro spin flipeado
		//int ene1=M[n]*vecinos;
		int ene1=-ene0;	

		// metropolis: aceptar flipeo solo si r < exp(-(ene1-ene0)/temp)
		float p=exp(-(ene1-ene0)/temp);

		// numero random entre [0,1] uniforme
		//float r=float(rand())/RAND_MAX;->philox
		float rn = uniform(n, seed, t);

		if(rn<p) M[n]*=-1;	
	}
};


// para imprimir una matriz LxL guardada en el HOST
void print_matrix(int *M, std::ofstream &fout, int L)
{
	for(int i=0;i<L;i++){ 
		for(int j=0;j<L;j++){ 
			fout << M[i*L+j] << " "; 
		}
		fout << "\n";
	}
	fout << "\n" << std::endl;
}


