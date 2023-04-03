// TODO significa "lo que hay que hacer" :-)

// TODO: reemplace este header por uno util_GPU.h
// que contenga las implementaciones paralelas de las funciones
// que se llaman en el main()
#include "util_CPU.h"

int main(int argc, char **argv)
{

	// primer argumento: tamanio lateral del sistema (default=128)
	int L=(argc>1)?(atoi(argv[1])):(128);
	if(L%2!=1 || L<0){
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

	// TODO: defina versio paralela de alocar_matriz(...) en util_GPU.h 
	// declara e inicializamos random la matriz M_{ij} = +-1 
	int *M = alocar_matriz(N);
	
	// TODO: defina version paralela inicializar_matriz_random(...) en util_GPU.h
	// inicializa M_{i,j} = +-1 en forma random
	//inicializar_matriz_random(M,N);
	inicializar_matriz_magnetizado(M,N);

	std::ofstream magnetizacion_out("magnetizacion.dat");

	#ifdef MOVIE
	std::ofstream evolucion_out("evolucion.dat");
	#endif

	// loop temporal
	for(int t=0;t<trun;t++){

		// TODO: escriba en util_GPU.h la version paralela de get_magnetizacion(....) 
		// equivalente a la escrita en util_CPU.h para host 
		magnetizacion_out << get_magnetizacion(M,N) << std::endl;

		// TODO: escriba la version paralela de metropolis(....) en util_GPU.h		

		// itera solo sobre los sitios pares	
		metropolis(M,temp,L,BLANCO);

		// itera solo sobre los sitios impares
		metropolis(M,temp,L,NEGRO);	

		// compile con nvcc -DMOVIE=numero para activar
		// donde numero define la separacion entre cuadros
		// para hacer una pelicula (ver readme.txt)
		// use con cuidado, no imprima archivos gigantescos...
		// note que el ultimo frame siempre se imprime
		#ifdef MOVIE
		if(t%MOVIE==0 || t==trun-1){
			print_matrix(M,evolucion_out,L);
		}
		#endif
	}

////////////////////////////////////////////////

	return 0;
}


