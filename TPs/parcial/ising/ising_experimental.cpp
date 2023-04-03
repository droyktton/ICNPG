// esta version es para jugar un poco mas...

#include "util_CPU_alternative.h"
#include "grafo.h"

int main(int argc, char **argv)
{

	// primer argumento: tamanio lateral del sistema (default=128)
	int L=(argc>1)?(atoi(argv[1])):(128);
	if(L%2!=0 || L<0){
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

	// TODO: aloque memoria en device
	// declaramos e inicializamos random la matriz M_{ij} = +-1 
	int *M = alocar_matriz(M,N);
	
	// TODO: inicialize en device
	// inicializar con -1,1 random uniformemente distribuido
	inicializar_matriz_random(M,N);
	//inicializar_matriz_magnetizado(M,N);
	//inicializar_matriz_pared(M,N);

	std::ofstream magnetizacion_out("magnetizacion.dat");
	std::ofstream evolucion_out("evolucion.dat");

	// loop temporal
	for(int t=0;t<trun;t++){

		// TODO: escriba una funcion get_magnetizacion() de device 
		magnetizacion_out << get_magnetizacion(M,N) << std::endl;

		//TODO: escriba una funcion de device metropolis
		// que opere sobre sitios pares o impares.

		// iteramos solo sobre los pares	
		metropolis(M,temp,L,BLANCO);

		// iteramos solo sobre los impares
		metropolis(M,temp,L,NEGRO);	

		#ifdef MOVIE
		if(t%MOVIE==0 || t==trun-1){
			imprimir_grafo_conectividad(M,L);
			print_matrix(M,evolucion_out,L);
		}
		#endif
	}

////////////////////////////////////////////////

	return 0;
}


