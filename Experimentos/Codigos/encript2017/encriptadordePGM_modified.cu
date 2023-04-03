#include<fstream>
#include<iostream>
#include<ctime>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
// TODO: agregue los headers que considere necesario...
// ....

#define L	200

// declaracion/definicion de funciones de lectura/escritura
#include "desencript.h"


// TODO: escriba su/sus functors o su/sus kernels aqui abajo...
#include "secreto.h" // eliminar del template


// TODO: complete la siguiente funcion de desencriptacion
void desencriptar(char *name)
{
	
	// numero total de pixels 
	int N=L*L;

        // declaramos vectores para la imagen encriptada, y para la desencriptada en el host de N=L*L pixels
	thrust::host_vector<int> h_enc(N); // imagen encriptada en host
	thrust::host_vector<int> h_des(N); // imagen desencriptada en host

	// lee la imagen encriptada en h_enc
	std::ifstream fin(name);	   // archivo de entrada
	leepgm(fin,h_enc);		   // lectura de imagen encriptada

        // Declaramos y alocamos vectores de device (de C o thrust) para la imagen encriptada y para la desencriptada
        // y hacemos la copia de la imagen encriptada de host a device
	thrust::device_vector<int> enc(h_enc);  
	thrust::device_vector<int> des(h_enc); 
	int *enc_raw=thrust::raw_pointer_cast(enc.data());
	int *des_raw=thrust::raw_pointer_cast(des.data());

        // TODO: declare y aloque vectores de device (thrust o C) para los offsets
	thrust::device_vector<int> offsets(L); 
	int *offsets_raw=thrust::raw_pointer_cast(offsets.data());

	// TODO: encuentre los offsets de cada fila en device
	for(int j=0;j<L;j++) offsets[j]=thrust::reduce(enc.begin()+j*L,enc.begin()+(j+1)*L); // calcule los offsets de cada fila
	// mejor es usar un reduce by key...

	// TODO: lance un kernel o algoritmo de thrust para desencriptar la imagen dados los offsets
	// (defina mas arriba sus kernels o functors...)
	// SOLUCION con Thrust
	/*thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(N),des.begin(),
		mifunctor(offsets_raw,enc_raw)
	);*/

	// SOLUCION con CUDA C
	mikernel<<<(N+255)/256, 256>>>(offsets_raw, enc_raw, des_raw);


	// Copiamos la imagen desencriptada de device a host en h_des 
	thrust::copy(des.begin(),des.end(),h_des.begin());

	// Escribe la imagen desencriptada h_des en un file pgm
	// si no desencripto imprimira la imagen original
	char nom[50]; sprintf(nom,"%s_desencripted.pgm",name);
	std::ofstream fout2(nom);
	escribepgm(fout2, h_des);
}


int main(int argc, char **argv)
{
	#ifdef ENCRIPTAR
	std::cout << "encriptando " << argv[1] << std::endl;
	encriptar(argv[1]);
	#else
	std::cout << "desencriptando " << argv[1] << std::endl;
	desencriptar(argv[1]);	
	#endif

	return 0;
}
