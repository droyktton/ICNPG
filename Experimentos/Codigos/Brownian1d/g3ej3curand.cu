/*
A.B. Kolton, 1 Nov 2013. Para ICNPG2013

Correr y comparar con la solucion analitica para T=0.75
gnuplot> V(x)=sin(2*pi*x) + 0.25*sin(4*pi*x); plot 'histogramas.dat' u 1:2 w lp, exp(-V(x)/0.75)/1.53694 lw 2

*/
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "histo.h"

#ifndef Dt
#define Dt		    0.0001      // paso de tiempo 
#endif

#define NROBINS		100        // numero de bins
#define NROPARTS	10000000    // numero de particulas	
#define SEED		123456789  // una semilla global 

// fuerza sobre una particula
__device__ float Fuerza(float x)
{
	// return -d/dx [sinf(2 M_PI x) + 0.25 sinf(4 M_PI x)]
	return -M_PI*(2.0f*cosf(2.0f*M_PI*x) + cosf(4.0f*M_PI*x));
}

// este functor hace evolucionar una particula browniana
// trun pasos a una temperatura T
struct dinamica
{
  // pasos de tiempo
  unsigned int trun; //numero de iteraciones a hacer entre histogramas
  float T;	      // temperatura
  float fac;         // una variable auxiliar
  int etapa;      // etapa para guardar datos
  dinamica(float _T, unsigned int _trun, int _etapa):T(_T),trun(_trun),etapa(_etapa)
  {
  	fac=sqrtf(2.0f*T*Dt);
  };	

  // tid es un contador que identifica la particula, y x es su posicion, 
  // a ser actualizada con esta funcion
  __device__
  float operator()(unsigned int tid, float x)
  {
    curandStatePhilox4_32_10_t s;    
    
    // seed a random number generator (unico para cada hilo)
    curand_init(SEED,tid, etapa, &s);

    // cada hilo genera un random walk de trun pasos
    for(unsigned int i = 0; i < trun; ++i)
    {      
      float NroGausiano=curand_normal(&s);

      // TODO: aplique un paso de evolucion temporal usando el numero gausiano, el campo de fuerza, y la variable auxiliar "fac"
      // como se indica en la Ec 3 de la Guia 3	
      x = x + Fuerza(x)*Dt + fac*NroGausiano;

      // TODO: fuerce las condiciones periodicas, tal que x este siempre en [0,1)
      if(x<0.0f) x+=1.0f;
      if(x>1.0f) x-=1.0f;	
    }
    return x; 	
  }
};

// si ponemos esto, nos evitamos los "thrust::"
using namespace thrust;	

int main(void)
{

  int card;
  cudaGetDevice(&card);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, card);
  std::cout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";

  // TODO declare un device_vector "X" para guardar las posiciones de las NROPARTS particulas
  device_vector<float> X(NROPARTS);

  // TODO Usando algoritmos de thrust inicialice las posiciones de las particulas: 
  // (1) Todas en el origen, (2) Uniforme en [0,1), (3) Cualquier otra que se le ocurra.	
  //fill(X.begin(),X.end(),float(0.0));	
  sequence(X.begin(),X.end(),float(0.0),float(1.0/NROPARTS)); 

  // TODO declare un device_vector "Histogram" para guardar el histograma normalizado
  // de las posiciones de las particulas, discretizada en NROBINS bins...
  device_vector<float> Histogram(NROBINS);
 
  float Temp=0.75f; // temperatura
  unsigned trun=100; // pasos entre histogramas	
  std::ofstream histout("histogramas.dat"); // file para guardar los histogramas

  for(int etapa=0;etapa<100;etapa++){
    // estas dos funciones calculan e imprimen la distribucion de probabilidad
    // de un array de floats en el intervalo [0,1] 
    dense_histogram_data_on_device(X,Histogram, float(0.0), float(1.0));  
    print_histograma(Histogram, float(0.0), float(1.0), histout);
     
     // TODO: examine el functor "dinamica()" de mas arriba, y complete los argumentos de 
  	 // transform adecuadamente: transform(........, dinamica(Temp,trun,tiempo));
     // AYUDA: counting_iterator<int>(0), counting_iterator<int>(NROPARTS) "simula" un rango 
  	 // de una secuencia. Ver ejemplo SimplePi de clase. 	 
  	 // solucion:
       transform(
        counting_iterator<int>(0),
        counting_iterator<int>(NROPARTS),
        X.begin(),
        X.begin(),
        dinamica(Temp,trun,etapa)
    );
  }   

  return 0;
}


