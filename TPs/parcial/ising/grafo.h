// matriz de conectividad
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/convert.h>
#include <cusp/graph/connected_components.h>
#include <cusp/array1d.h>
#include <cstdlib>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <thrust/iterator/constant_iterator.h>
#include "histograma.h"

typedef int   IndexType;
typedef int ValueType;
typedef cusp::host_memory MemorySpace;
typedef cusp::csr_matrix<IndexType, ValueType, MemorySpace> GraphType;
typedef cusp::coo_matrix<IndexType, ValueType, MemorySpace> COOType;

std::ofstream hisout("histogram.dat");
std::ofstream cumout("cumulative.dat");
std::ofstream compout("components.dat");


//void llenar_grafo_conectividad(GraphType &G, int L)
int imprimir_grafo_conectividad(int *M,int L)
{

	int N=L*L;

	std::vector<thrust::tuple<float,float> > vec;
	thrust::tuple<float,float> tup;
	for(int n=0;n<N;n++){ 

		/* esta buggy, corregido abajo 
		// los indices de los sitios vecinos
		// con condiciones de contorno periodicas
		int izq=(n-1+N)%N;
		int der=(n+1)%N;
		int arr=(n+L)%N;
		int aba=(n-L+N)%N;
		*/
		
		/* vecinos en diagonal...
		int nx=n%L;
		int ny=int(n/L);

		// los indices de los sitios vecinos
		// con condiciones de contorno periodicas
		int izq = ((nx-1+L)%L) + ((ny-1+L)%L)*L ; // noroeste 
		int der = ((nx+1+L)%L) + ((ny-1+L)%L)*L ; // noreste
		int arr = ((nx-1+L)%L) + ((ny+1+L)%L)*L ; // sudoeste
		int aba = ((nx+1+L)%L) + ((ny+1+L)%L)*L ; // sudeste	
		*/

		int nx=n%L;
		int ny=int(n/L);

		// los indices de los sitios vecinos
		// con condiciones de contorno periodicas
		int izq = ((nx-1+L)%L) + ny*L ;
		int der = ((nx+1)%L) + ny*L ;
		int aba = nx + ((ny+1)%L)*L ;
		int arr = nx + ((ny-1+L)%L)*L ;

		if(1)
		{
		thrust::get<0>(tup)=n;

		thrust::get<1>(tup)=n;
		if(M[n]==M[n]) vec.push_back(tup);		

		thrust::get<1>(tup)=izq;
		if(M[n]==M[izq]) vec.push_back(tup);		

		thrust::get<1>(tup)=der;
		if(M[n]==M[der]) vec.push_back(tup);		

		thrust::get<1>(tup)=aba;
		if(M[n]==M[aba]) vec.push_back(tup);		

		thrust::get<1>(tup)=arr;
		if(M[n]==M[arr]) vec.push_back(tup);		
		}
	}
	std::cout << "edges = " << vec.size() << std::endl;

	COOType A(N,N,vec.size());
	for(int n=0;n<vec.size();n++){ 
		int row=thrust::get<0>(vec[n]);				
		int col=thrust::get<1>(vec[n]);			
		A.row_indices[n]=row; A.column_indices[n]=col; A.values[n]=1;
	}
	GraphType G(A);
	
	cusp::array1d<IndexType,MemorySpace> components(G.num_rows);
    	size_t num_components = cusp::graph::connected_components(G, components);
	std::cout << "number of components = " << num_components << std::endl;

	int *H = components.data();
	print_matrix(H,compout,L);
	//print_matrix(H,H,compout,L);

	
	// agrupo los vertices de una misma componente
	thrust::sort(components.begin(),components.end());

	// seria mejor no perder info la componente en la que estan los vertices 
	//thrust::sort_by_key(components.begin(),components.end(),vertices.begin());

	//cusp::print(components);	
	thrust::host_vector<int> sizes(G.num_rows);
	thrust::host_vector<int> keys(G.num_rows);

	typedef thrust::host_vector<int>::iterator int_it;

	// calculo el tamanio de cada componente
	thrust::pair<int_it,int_it> new_end;
	new_end = 
	thrust::reduce_by_key
	(
		components.begin(),components.end(),
		thrust::make_constant_iterator(1),
		keys.begin(),sizes.begin() 
	);
	int m=int(new_end.second-sizes.begin());
	std::cout << "clusters=" << m << std::endl;
	// keys[i] tiene el numero de componente del cluster de tamanio size[i], i=0,..,nclusters-1
	// 

	// metodo 1: sparse histogram
	thrust::host_vector<int> histogram_values;
    	thrust::host_vector<int> histogram_counts;
	sparse_histogram(sizes,histogram_values,histogram_counts);
	for(int i=0;i<histogram_values.size();i++)
	hisout << histogram_values[i] << " " << histogram_counts[i] << std::endl;
	hisout << "\n" << std::endl;

	// metodo 2: cumulated histogram
	// tamanio de cada componente, por orden...
	cusp::array1d<IndexType,MemorySpace> ordered_sizes(sizes.begin(),sizes.begin()+m);
	thrust::sort(ordered_sizes.begin(), ordered_sizes.end());
	for(int i=0;i<ordered_sizes.size();i++)
	cumout << ordered_sizes[i] << std::endl;
	cumout << "\n" << std::endl;
 		
	return m;

	//std::cout << A.num_entries << " " << G.num_entries << std::endl;
	//std::cout << "mayor componente = " << sizes[m-1] << std::endl; 

}

