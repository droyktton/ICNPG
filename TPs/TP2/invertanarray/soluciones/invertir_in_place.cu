#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <cstdlib>

#define N	10

// numeros random del 0 al 20
int mirand()
{
	return (rand()%20);
}

void print_vector(thrust::host_vector<int> H)
{
	for(int i=0;i<N;i++)
	std::cout << H[i] << " ";
	std::cout << std::endl;
}

int main()
{
	thrust::host_vector<int> h(N);
	thrust::generate(h.begin(),h.end(),mirand);

	std::cout << "array original\n";
	print_vector(h);


	// out of place:

	thrust::host_vector<int> hi(h.rbegin(),h.rend());
	std::cout << "array invertido out of place\n";
	print_vector(hi);
	// usamos el copy constructor de hi a partir del rango definido por iteradores marcha atras...
	// se podria haber hecho un thrust::copy o un thrust::transform, etc.

	

	// inplace:

	thrust::swap_ranges(h.rbegin(),h.rbegin()+N/2,h.begin());
	std::cout << "array invertido in place\n";
	print_vector(h);
	// usamos swap ranges con el mismo vector, pero usando solo la mitad de su rango 
	// para no swapear dos veces el mismo par de elementos.

}
