/*
A.B. Kolton, 1 Nov 2013. Para ICNPG2013
*/
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "histo.h"

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // un counter-based RNG

// esta funcion retorna un numero aleatorio gaussiano 
// con media 0 y variancia 1, dado un numero generado
// por philox
__device__
float box_muller(RNG::ctr_type r_philox)
{
	// transforma el philox number a dos uniformes en (0,1]
 	float u1 = u01_open_closed_32_53(r_philox[0]);
  	float u2 = u01_open_closed_32_53(r_philox[1]);

  	float r = sqrtf( -2.0*logf(u1) );
  	float theta = 2.0*M_PI*u2;
	return r*sinf(theta);    			
}


#define Dt		0.001
#define NROBINS		100   // numero de bins
#define NROPARTS	500000 // numero de particulas	
#define SEED		123456789 // una semilla global 

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
  unsigned int t;    // numero de iteraciones total ~ tiempo 
  float T;	      // temperatura
  float fac;         // una variable auxiliar
  dinamica(float _T, unsigned int _trun, unsigned int _t):T(_T),trun(_trun),t(_t)
  {
	fac=sqrtf(2.0f*T*Dt);
  };	

  // tid es un contador que identifica la particula, y x es su posicion, 
  // a ser actualizada con esta funcion
  __device__
  float operator()(unsigned int tid, float x)
  {
    // keys and counters 
    RNG philox; 	
    RNG::ctr_type c={{}};
    RNG::key_type k={{}};
    RNG::ctr_type r;

    // TODO:Elija k[0] tal que se genere una secuencia random "unica" para cada particula	
    k[0]=tid; 

    // TODO:Elija c[1] tal que se generen secuencias random distintas para cada corrida cambiando SEED	
    c[1]=SEED;

    for(unsigned int i = 0; i < trun; ++i)
    {
      // TODO: Elija c[0] tal que -cada particula- tenga una secuencia random temporalmente descorrelacionada	
      c[0]=t+i; 

      // una llamada retorna dos numeros random descorrelacionados
      // empaquetados en un RNG::ctr_type. Escriba la llamada al RNG "philox"	
      // Consulte Simple_Pi visto en clase	
      r = philox(c, k); 

      // convierte r en un numero gaussiano N[0,1], usando la funcion box_muller
      // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
      float NroGausiano=box_muller(r);

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

#include <omp.h>

// si ponemos esto, nos evitamos los "thrust::"
using namespace thrust;	

int main(void)
{

  #ifdef OMP
  // imprime el maximo numero de threads en el que correra la version openMP
  std::cout << "#host OMP threads = " << omp_get_max_threads() << std::endl;
  #else
  int card;
  cudaGetDevice(&card);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, card);
  std::cout << "\nDevice Selected " << card << " " << deviceProp.name << "\n";
  #endif			

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
  unsigned trun=1; // pasos entre histogramas	
  unsigned tiempo=0; // tiempo absoluto

  std::ofstream histout("histogramas.dat"); // file para guardar los histogramas

  for(unsigned n=0;n<1000;n++,tiempo+=trun)
  {
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
  	      	dinamica(Temp,trun,tiempo)
  	 );
  }
  return 0;
}


