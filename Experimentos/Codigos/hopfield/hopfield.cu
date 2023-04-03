#define NMEM	3
#include "util_GPU.h"
#include "synapsis.h"

int main(int argc, char **argv)
{

	// primer argumento: tamanio lateral del sistema (default=128)
	int L=(argc>1)?(atoi(argv[1])):(128);
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

	// TODO: defina versio paralela de alocar_matriz(...) en util_GPU.h 
	// declara e inicializamos random la matriz M_{ij} = +-1 
	//int *M = alocar_matriz(N);
	
	thrust::host_vector<int> M_h(N,1);
	thrust::device_vector<int> M(M_h);
	int *M_ptr=thrust::raw_pointer_cast(&M[0]);
	thrust::transform
	(
		thrust::make_counting_iterator(0),thrust::make_counting_iterator(N), 
		M.begin(), 
		initialize_random(seed)
	);

	thrust::device_vector<bool> U(N,bool(0));
	bool *U_ptr=thrust::raw_pointer_cast(&U[0]);

	thrust::device_vector<int> W(N*NMEM,int(1));
	int *W_ptr=thrust::raw_pointer_cast(&W[0]);
	//llenar_memorias_from_file(W,L);
	llenar_memorias(W,L);

	// TODO: defina version paralela inicializar_matriz_random(...) en util_GPU.h
	// inicializa M_{i,j} = +-1 en forma random

	std::ofstream magnetizacion_out("magnetizacion.dat");

	#ifdef MOVIE
	std::ofstream evolucion_out("evolucion.dat");
	#endif

	float dtemp=temp/trun;

	// loop temporal
	for(int t=0;t<trun;t++){

		std::cout << "temp=" << temp << ", t=" << t << std::endl;
		
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

		// TODO: escriba en util_GPU.h la version paralela de get_magnetizacion(....) 
		// equivalente a la escrita en util_CPU.h para host 
		magnetizacion_out << float(thrust::reduce(M.begin(),M.end()))/N << std::endl;

		// TODO: escriba la version paralela de metropolis(....) en util_GPU.h		

		thrust::for_each(
			thrust::make_counting_iterator(0),thrust::make_counting_iterator(N),
			metropolitor(M_ptr,W_ptr,U_ptr,temp,L,t,seed));	

		thrust::transform_if(M.begin(),M.end(),U.begin(),M.begin(),thrust::negate<int>(),thrust::identity<bool>());

		temp-=dtemp;
	}

////////////////////////////////////////////////

	return 0;
}

// g++ -O2 -x c++ ising.cu -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I /usr/local/cuda-5.5/include/
