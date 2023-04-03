#include "histograma_thrust.h"

// una funcioncita para llenar un vector float de prueba
void llenar_exponential_deviate(thrust::device_vector<float> &);

int main(void)
{

  	const int N = 4000; // tamaño de tu input instantaneo
  	const int numbins=20; // numero de bins
	const float min=0.0; // minimo valor histogrameable
	const float max=3.0; // maximo valor histogrameable

	// crea el histogramador denso (el sparse para luego...)
  	histograma_denso H(numbins,min, max, N);

  	// llenamos con algo el float input
  	srand(123456);
 	thrust::device_vector<float> input(N);

	// acumulamos 100 samples
	for(int n=0;n<100;n++){
		llenar_exponential_deviate(input);
		
		// (1) le puedo dar un thrust::vector
  		//H.compute_dense_histogram(input);

		// (2) o un puntero a device crudeli, pero con su tamaño
		float *crudeli=thrust::raw_pointer_cast(input.data());
  		H.compute_dense_histogram(crudeli,N);
	}

	// el resultado en un file
  	std::ofstream fout("out.dat");	
  	H.print(fout);

	// para graficarlo lueguito:
	system("gnuplot -p -e \"set tit ' histograma de -log(rand)';plot 'out.dat' u 1:2 w lp t 'acumulado', '' u 1:3 w lp t 'instantaneo';\" ");

  	return 0;
}




void llenar_exponential_deviate(thrust::device_vector<float> &input)
{
	int N=input.size();
  	for(int i = 0; i < N; i++)
  	{
      		input[i] = -log(rand()/float(RAND_MAX));
		#ifdef DEBUG
      		std::cout << input[i] << " ";
		#endif
  	}
	#ifdef DEBUG
  	std::cout << std::endl;
	#endif
}


