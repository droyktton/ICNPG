/*

	Final B, ICNPG 2017: Hashblock

	Compilar:
	--------
	nvcc hashblock.cu  -o hashblock

	Correr:
	------

	El código se ejecuta como:

			./hashblock path_del_archivo/nombre_del_archivo 
	

	donde 
	
	nombre_del_archivo: es alguno de los archivos disponibles 
						de encabezados de bloques, elija alguno en el directorio blocks.

	Ej: ./hashblock block/block.4719xx 
	
	Agréguelo en su submit.sh preferido para correr en el cluster.

*/	
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "auxiliar.h"
#include <iostream>
#include "crc.h"

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
	printf("Reading header %s\n",filename );
	fscanf(fp,"%u\n",&(header->version));

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
	}
	fgetc(fp);

	hex2bin(header->prev_block,buffer);
	//hexdump(header->prev_block, HASHSIZE);

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
	}

	hex2bin(header->merkle_root,buffer);
	//hexdump(header->merkle_root, HASHSIZE);

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
__device__ __host__ void set_nonce(unsigned char *header, unsigned int nonce){

	header[76] = (nonce >> 1) & 0xff;
	header[77] = (nonce >> 2) & 0xff;
	header[78] = (nonce >> 3) & 0xff;
	header[79] = (nonce >> 4) & 0xff;
}


// Esta function retorna TRUE=1/FALSE=0 si el encabezado de bloque modificado con el nonce es valido/invalido.
__device__ __host__ 
bool validator(unsigned char *block)
{
	return bool(crcSlow(block, HEADER_BLOCK_SIZE) < DIFFICULTY); 		
}


//TODO: Agregue aqui los headers .h que considere necesarios
// #include ....
// #include ....


//TODO: defina aqui sus kernels y/o functors necesarios para sus tareas

/*
	Posible interface para un kernel de CUDA, puede modificarla si así lo desea
*/
__global__ void mineblock_kernel( unsigned char * header_inicial, bool *isvalid)
{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;

//	TODO: Complete el kernel que recibe el encabezado de bloque como un
//		  array  "header_inicial" y devuelve un array de bools "isvalid" 
//		  con el resultado de la validacion del mismo 
//	........

}

/*
	Posible interface para un functor de thrust, puede modificarla si así lo desea
*/
struct mineblock_functor
{
	unsigned char *header_inicial;
	mineblock_functor(unsigned char * block_):header_inicial(block_){};

	__device__ 
	bool operator()(int index)
	{
//	TODO: Complete el functor que recibe el encabezado de bloque como un
//		  array  "header_inicial" y devuelve un array de bools "isvalid" 
//		  con el resultado de la validacion del mismo 
//	........

		bool isvalid;
	//	....;
		return isvalid;
	}
};


/*
	Main 

	El ejecutable se llama como ./hashblock nombre_del_archivo
	donde 
	nombre_del_archivo: es alguno de los archivos disponibles 
						de header de bloques, incluido el path.

	Ej: ./hashblock block/block.4719xx
*/

int main(int argc, char **argv){

	// Declaramos un header de bloque.
	block_header header_struct;

    	/*
		El header se transforma a un array "header_inicial" de caracteres para que sea
		más sencillo de manejar en GPU. Se hace simplemente un cast.
     	*/
	unsigned char *header_inicial = (unsigned char *)&header_struct;

	/*
		Procesamos el input, leemos el encabezado de bloque desde archivo
	*/
    	if (argc == 2){
    		read_block_header(argv[1],&header_struct);
	
		// Muestra el header leido		
		std::cout << "----------------------------------" << std::endl;	
		std::cout << "Header struct Leido:" << std::endl;	
		print_block_header(header_struct);
		
		 // Mostramos el bloque como un array.
		printf("header in hexa: "); 
		hexdump(header_inicial, HEADER_BLOCK_SIZE);
		std::cout << "----------------------------------" << std::endl;	
    	}	
    	else
    	{
    		printf("Usage: ./hashblock file\n");
    		exit(1);
    	}
	std::cout << "inicializacion lista. Validando block header .... "; 


	// =========== AQUI EMPIEZA SU TAREA DE VALIDACION ================: 




	// TODO: Copie los datos al device usando todos los arrays de device o  
	//		 vectores de Thrust adecuados que necesite.
	
	//  TODO: Llame al kernel, o al algoritmo de Thrust que sea necesario para hacer la cantidad 
	//		de tests necesarios para validar el header del block leido


	int primer_indice_valido = -1; // por default es invalido
	// TODO: entre todos los tests realizados debe encontrar encontrar uno que valide el header leido
	//	 y asignar el valor correspondiente a "primer_indice_valido" 
	// primer_indice_valido = ......;
	
	
	// =========== AQUI TERMINA SU TAREA DE VALIDACION ================: 
	

	// check de validacion 
	if(primer_indice_valido!=-1)
	{

		std::cout << "listo\n\n" << std::endl; 
		set_nonce(header_inicial, primer_indice_valido);
		if(validator(header_inicial)){ 
			std::cout << "\n\n\n bien ahi!, se gano un ICNPGcoin, su indice valida el header leido:" << std::endl;
			
			// Imprime el encabezado de bloque validado, y el nonce obtenido
			hexdump((unsigned char*)&header_struct, sizeof(block_header));
			printf("Nonce = %d\n", header_struct.nonce);
			std::cout << "Envie su solucion y pase a cobrar su ICNPG-coin!" << std::endl;
		}
		else{		
			std::cout << "su indice no valida el header. Siga intentandolo." << std::endl;		
		} 		
	}
	else{
		std::cout << "Ud no hizo nada.\nDebe obtener un indice de validacion para el header leido. Suerte con eso!" << std::endl;
	}

	return 0;
}
