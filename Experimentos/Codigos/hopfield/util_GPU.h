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
#include<thrust/transform.h>
#include<thrust/functional.h>

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

struct initialize_random
{
	int seed;
	initialize_random(int _seed):seed(_seed){};

	__device__
	int operator()(int i)
	{
		float rn = uniform(i, seed, 0);
		return (rn>0.5)?(1):(-1);
	}	
};

struct metropolitor
{
	int *M;
	int *W;
	bool *U;
	float temp;
	int L;
	int shift;
	int N;
	int t;
	int seed;

	metropolitor(int *_M, int *_W, bool *_U, float _temp, int _L, int _t, int _seed):
	M(_M),W(_W),U(_U),temp(_temp),L(_L),t(_t), seed(_seed)
	{N=L*L;};	

	__device__
	void operator()(int i)
	{
		float vecinos=0;
		for(int j=0;j<N;j++){
			if(i!=j)
			for(int p=0;p<NMEM;p++){
				vecinos+=(W[i+p*N]*W[j+p*N]);
			}
			vecinos*=float(M[j])/NMEM;
		}
		// contribucion de nuestro spin sin flipear a la energia  
		float ene0=-M[i]*vecinos;	

		// contribucion a la energia de nuestro spin flipeado
		//int ene1=M[i]*vecinos;
		float ene1=-ene0;	

		// metropolis: aceptar flipeo solo si r < exp(-(ene1-ene0)/temp)
		float p=exp(-(ene1-ene0)/temp);

		float rn = uniform(i, seed, t);

		if(rn<p) U[i]=1;//flip
		else U[i]=0;//no flip	
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


