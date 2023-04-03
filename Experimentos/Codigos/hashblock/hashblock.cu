/*

	Final B, ICNPG 2017: Hashblock

	Compilar:
	--------
	nvcc crc.cu hashblock.cu  -o hashblock

	Correr:
	------

	El código se ejecuta como:

			./hashblock nombre_del_archivo [nonce_inicial]
	

	donde 
	
	nombre_del_archivo: es alguno de los archivos disponibles 
						de encabezados de bloques, elija alguno en el directorio blocks.

	nonce_inicial: (opcional) es el nonce que se usará para empezar a
				   buscar. Si no se incluye, nonce_inicial es cero.

	Ej: ./gashblock block/block.4719xx 23

	
	Agréguelo en su submit.sh preferido para correr en el cluster.

*/	
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "common/curso.h"
#include "auxiliar.h"
#include "crc.cuh"

//
// Definimos una dificultad arbitraria
//
#define HEADER_BLOCK_SIZE 80
#define DIFFICULTY 2060

/*
	Este es el tipo de estructura para los header de bloques
*/
typedef struct block_header {
	unsigned int	version; 			  	// Version del blockchain
	unsigned char	prev_block[HASHSIZE];   // Hash del bloque anterior
	unsigned char	merkle_root[HASHSIZE];  // Hash del arbol de Merkle
	unsigned int	timestamp;				// Fecha y hora
	unsigned int	bits;          			// Tamaño del bloque
	unsigned int	nonce;					// nonce
} block_header;


/*
	Función que lee un header de bloque.
*/
void read_block_header(char* filename, block_header *header){

	unsigned char buffer[64];
	FILE *fp = fopen(filename,"r");
	printf("Reading block %s\n",filename );
	fscanf(fp,"%u\n",&(header->version));

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
	}
	fgetc(fp);

	hex2bin(header->prev_block,buffer);
	hexdump(header->prev_block, HASHSIZE);

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
	}

	hex2bin(header->merkle_root,buffer);
	hexdump(header->merkle_root, HASHSIZE);

	fscanf(fp,"%u\n",&(header->timestamp));
	fscanf(fp,"%u\n",&(header->bits));
	// the endianess of the checksums needs to be little, this swaps them form the big endian format you normally see in block explorer
	byte_swap(header->prev_block, HASHSIZE);
	byte_swap(header->merkle_root, HASHSIZE);
}

/*
	Muestra un bloque por pantalla
*/

void print_block_header(block_header header){

	printf("Version     : %u\n",header.version);
	printf("prev_block  : ");
	hexdump(header.prev_block, HASHSIZE);
	printf("merkle_root : ");
	hexdump(header.merkle_root, HASHSIZE);
	printf("timestamp   : %u\n",header.timestamp);
	printf("bits   	    : %u\n",header.bits);

}

/*
	Esta función modifica el bloque cambiando el nonce en los bytes 76 a 79
*/
__device__ void set_nonce(unsigned char *block, unsigned int nonce){

	block[76] = (nonce >> 1) & 0xff;
	block[77] = (nonce >> 2) & 0xff;
	block[78] = (nonce >> 3) & 0xff;
	block[79] = (nonce >> 4) & 0xff;
}



/*
	Posible interface para un kernel de CUDA, puede modificarla si así lo desea
*/

__global__ void mineblock( unsigned char * block, unsigned int *hashes)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
//	TODO: Complete el kernel que recibe el encabezado de bloque como un
//		  array y devuelve un array con los hashes de cada encabezado modificado 
// 		  de acuerdo a cada nonce.




}



/*
	Main 

	El ejecutable se llama como ./gashblock nombre_del_archivo [nonce_inicial]
	donde 
	nombre_del_archivo: es alguno de los archivos disponibles 
						de header de bloques.

	nonce_inicial: (opcional) es el nonce que se usará para empezar a
				   buscar. Si no se incluye, nonce_inicial es cero.

	Ej: ./gashblock block/block.4719xx 23

*/

int main(int argc, char **argv){

	// Declaramos un header de bloque.
	block_header header;

	// nonce inicial default
	unsigned int nonce_init = 0;	
	/*
		Procesamos el input, leemos el encabezado de bloque desde archivo
	*/
    if (argc == 2){
    	read_block_header(argv[1],&header);
    	nonce_init = 0;
    }
    else if(argc == 3){
        read_block_header(argv[1],&header);
       	nonce_init = atoi(argv[2]);
    }else{
    	printf("Usage: ./hashblock file [nonce]\n");
    	exit(1);
    }
    // Muestra el header leido
    print_block_header(header);
    /*
		El header se transforma a un array de caracteres para que sea
		más sencillo de manejar en GPU.
     */

	unsigned char block[HEADER_BLOCK_SIZE], *pblock;
	pblock = (unsigned char *)&header;
	printf("Block in hexa: ");  // Mostramos el bloque como un array.
	hexdump(pblock, HEADER_BLOCK_SIZE);

	// TODO: Copie los datos al device o utilice los 
	//		 vectores de Thrust adecuados.


	//
	//	TODO: Dimensione los bloques de hilos, o utilice las funciones 
	//		  de thrust necesarias.
	//
	int nThreads;
	dim3 threads(nThreads);
	dim3 blocks((nk+nThreads-1)/nThreads);
	printf("blocks:%d\n",blocks.x );

	//  TODO: Llame al kernel, o a los transforms de Thrust que sean necesarios
	


	// Imprime el encabezado de bloque validado, y el nonce obten

	printf("Block header : ");
	hexdump((unsigned char*)&header, sizeof(block_header));
	printf("Nonce = %d\n", header.nonce);
	return(0);


}
