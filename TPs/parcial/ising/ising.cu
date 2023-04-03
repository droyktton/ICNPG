// COMPILAR ASI PARA IMPRIMER SNAPHOTS CADA 100 pasos
// nvcc -DMOVIE=100 -I. ising.cu

// SINO ASI:
// nvcc -I. ising.cu

#include "util_GPU.h"

int main(int argc, char **argv)
{

	// primer argumento: tamanio lateral del sistema (default=128)
	int L=(argc>1)?(atoi(argv[1])):(128);
	if(L%2==1 || L<0){
		std::cout << "error: L debe ser par positivo" << std::endl;
		exit(1);	
	}	
	int N=L*L;

	// segundo argumento: temperatura (default=2.0)
	float temp=(argc>2)?(atof(argv[2])):(2.0);

	// tercer argumento: iteraciones totales (default=100)
	int trun=(argc>3)?(atoi(argv[3])):(100);

	// cuarto argumento: semilla global (default=0)
	int seed = (argc>4)?(atoi(argv[4])):(0);
	srand(seed);


	std::cout << "L="    << L    << std::endl;
	std::cout << "temp=" << temp << std::endl;
	std::cout << "trun=" << trun << std::endl;
	std::cout << "seed=" << seed << std::endl;


/////// DESDE AQUI HACER TODO EN DEVICE ///////

	thrust::host_vector<int> M_h(N,1);
	thrust::device_vector<int> M(M_h);
	int *M_ptr=thrust::raw_pointer_cast(&M[0]);

	std::ofstream magnetizacion_out("magnetizacion.dat");

	#ifdef MOVIE
	std::ofstream evolucion_out("evolucion.dat");
	#endif

	// loop temporal
	for(int t=0;t<trun;t++){
		magnetizacion_out << float(thrust::reduce(M.begin(),M.end()))/N << std::endl;

		int npares=N/2;
		thrust::for_each(
			thrust::make_counting_iterator(0),thrust::make_counting_iterator(npares),
			metropolitor(M_ptr,temp,L,PARES,t,seed));	

		int nimpares=N/2;
		thrust::for_each(
			thrust::make_counting_iterator(0),thrust::make_counting_iterator(nimpares),
			metropolitor(M_ptr,temp,L,IMPARES,t,seed)
		);	

		// compile con nvcc -DMOVIE=numero para activar
		// donde numero define la separacion entre cuadros
		// para hacer una pelicula (ver readme.txt)
		// use con cuidado, no imprima archivos gigantescos...
		// note que el ultimo frame siempre se imprime
		#ifdef MOVIE
		if(t%MOVIE==0 || t==trun-1){
			thrust::copy(M.begin(),M.end(),M_h.begin());
			print_matrix(M_h.data(),evolucion_out,L);
		}
		#endif
	}

////////////////////////////////////////////////

	return 0;
}

// g++ -O2 -x c++ ising.cu -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I /usr/local/cuda-5.5/include/
