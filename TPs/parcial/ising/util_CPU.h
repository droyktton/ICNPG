#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdlib>

int * alocar_matriz(int N)
{
	return (int *)malloc(sizeof(int)*N);	
}

#define PARES	0
#define IMPARES	1
#define BLANCO	0
#define NEGRO	1
void metropolis(int *M, float temp, int L, int shift)
{
	int N=L*L;
	int n=shift;
	for(int n=shift;n<N;n+=2) // L debe ser par 
	{ 
		//
		int nx=n%L;
		int ny=int(n/L);

		//if((nx+ny)%2==shift)
		{

		// los indices de los sitios vecinos
		// con condiciones de contorno periodicas
		/*int izq = ((nx-1+L)%L) + ((ny-1+L)%L)*L ; // noroeste 
		int der = ((nx+1+L)%L) + ((ny-1+L)%L)*L ; // noreste
		int arr = ((nx-1+L)%L) + ((ny+1+L)%L)*L ; // sudoeste
		int aba = ((nx+1+L)%L) + ((ny+1+L)%L)*L ; // sudeste	
		*/

		// con condiciones de contorno periodicas en y pero helicas con paso 1 en X...
		int izq=(n-1+N)%N;
		int der=(n+1)%N;
		int arr=(n+L)%N;
		int aba=(n-L+N)%N;

		// la magnetizacion total de los vecinos
		int vecinos=M[izq]+M[der]+M[arr]+M[aba];
		
		// contribucion de nuestro spin sin flipear a la energia  
		int ene0=-M[n]*vecinos;	

		// contribucion a la energia de nuestro spin flipeado
		int ene1=M[n]*vecinos;	

		// metropolis: aceptar flipeo solo si r < exp(-(ene1-ene0)/temp)
		float p=exp(-(ene1-ene0)/temp);
		float r=float(rand())/RAND_MAX;
		if(r<p) M[n]*=-1;
		}
	}
}


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


// inicializa la matriz
void inicializar_matriz_random(int *M, int N)
{
	float r;
	for(int i=0;i<N;i++) 
	{
		r = float(rand())/RAND_MAX;
		M[i]=(r>0.5)?(1):(-1);
	}
}

// inicializa la matriz
void inicializar_matriz_magnetizado(int *M, int N)
{
	for(int i=0;i<N;i++) 
	{
		M[i]=1;
	}
}

// retorna la magnatizacion
float get_magnetizacion(int *M, int N)
{
	float sum=0.0;
	for(int i=0;i<N;i++) sum+=M[i];
	return sum*=1.0/N;
}





