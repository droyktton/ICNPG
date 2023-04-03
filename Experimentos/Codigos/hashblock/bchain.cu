/*************************** HEADER FILES ***************************/
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
	//printf("Version    : %u\n",header->version);

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
		//printf("%c", buffer[i]);
	}
	fgetc(fp);

	//printf("\n");
	//printf("buffer : ");
	//hexdump(buffer, HASHSIZE);

	hex2bin(header->prev_block,buffer);
	//printf("prev_block : ");
	hexdump(header->prev_block, HASHSIZE);

	for (int i = 0; i < 64; ++i)
	{
		buffer[i] = (unsigned char)fgetc(fp);
	//	printf("%c", buffer[i]);
	}
	//printf("\n");
	//printf("buffer : ");
	//hexdump(buffer, HASHSIZE);

	hex2bin(header->merkle_root,buffer);
	//printf("merkle_root : ");
	hexdump(header->merkle_root, HASHSIZE);

	fscanf(fp,"%u\n",&(header->timestamp));
	//printf("timestamp    : %u\n",header->timestamp);

	fscanf(fp,"%u\n",&(header->bits));
	//printf("bits         : %u\n",header->bits);	
	
	// the endianess of the checksums needs to be little, this swaps them form the big endian format you normally see in block explorer
	byte_swap(header->prev_block, HASHSIZE);
	byte_swap(header->merkle_root, HASHSIZE);
}

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

void set_nonce(unsigned char *block, unsigned int nonce){

	block[76] = (nonce >> 1) & 0xff;
	block[77] = (nonce >> 2) & 0xff;
	block[78] = (nonce >> 3) & 0xff;
	block[79] = (nonce >> 4) & 0xff;
}








__global__ void mineblock_simple( unsigned char * block, unsigned int difficulty, unsigned int nonce)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = index + nonce;
	unsigned int hash = 0;

	unsigned char  lblock[HEADER_BLOCK_SIZE];
 	//printf("thread: %d   hash:%u\n",idx,block[77]);
	//header->nonce = idx;

	// for (size_t n = 76; n < HEADER_BLOCK_SIZE; n++){
	// 	lblock[n] = block[n];
	//  	printf("%.2x",lblock[n]); 
	// }
	//printf("\n");	 	    	
	lblock[79] = (idx >> 1) & 0xff;
	lblock[78] = (idx >> 2) & 0xff;
	lblock[77] = (idx >> 3) & 0xff;
	lblock[76] = (idx >> 4) & 0xff;
	//printf("%.2x %.2x %.2x %.2x\n", block[76],block[77],block[78],block[79]);
	hash = crcSlow(lblock, HEADER_BLOCK_SIZE);	

	__syncthreads();
	if(hash<difficulty)
		printf("thread: %d   hash:%u\n",idx,hash );
}


__global__ void mineblock_shared( unsigned char * block, unsigned int difficulty, unsigned int *nonce)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int idx = index;
	extern __shared__ unsigned int hashes[];

	unsigned char  lblock[HEADER_BLOCK_SIZE];
 	//printf("thread: %d   hash:%u\n",idx,block[77]);
	//header->nonce = idx;
	// Copio a un array local a cada thread
	for (int n = 0; n < HEADER_BLOCK_SIZE; n++){
		lblock[n] = block[n];
	 	//printf("%.2x",lblock[n]); 
	}
	//printf("\n");	 	    	
	lblock[76] = (idx >> 1) & 0xff;
	lblock[77] = (idx >> 2) & 0xff;
	lblock[78] = (idx >> 3) & 0xff;
	lblock[79] = (idx >> 4) & 0xff;
	//printf("%.2x %.2x %.2x %.2x\n", block[76],block[77],block[78],block[79]);
	hashes[tid] = crcSlow(lblock, HEADER_BLOCK_SIZE);	

	// if(hash<difficulty)
	// 	printf("thread: %d   hash:%u\n",index,hashes[tid] );
	__syncthreads();
	// get minimum nonce across block
    int i = blockDim.x/2;
    while (i != 0) {
        if (tid < i)
            hashes[tid] = hashes[tid]<hashes[tid + i]?hashes[tid]:hashes[tid + i];
        __syncthreads();
        i /= 2;
    }

    if(tid==0)
        nonce[blockIdx.x]=hashes[0];


}



/*
	Main del block hashing
	Se llama como ./gashblock nombre_del_archivo [nonce_inicial]
	donde 
	nombre_del_archivo: es alguno de los archivos disponibles 
						de header de bloques.

	nonce_inicial: (opcional) es el nonce que se usará para 
				   buscar.

	Ej: ./gashblock block.47191 23

*/

int main(int argc, char **argv){

	// Declaramos un header de bloque, y un puntero en el device.
	block_header header;
	// TODO: Declare las variables en el device aqui.
	block_header *d_header;

	// nonce inicial default
	unsigned int nonce_init = 0;	
	//
	// read block_header
	//
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
	printf("Block in hexa: ");
	hexdump(pblock, HEADER_BLOCK_SIZE);

	// TODO: Copie los datos al device o utilice los 
	//		 vectores de Thrust adecuados.

	unsigned char *d_block;
	HANDLE_ERROR(cudaMalloc((void**)&d_block,HEADER_BLOCK_SIZE*sizeof(char)));
	HANDLE_ERROR(cudaMemcpy(d_block,pblock,HEADER_BLOCK_SIZE*sizeof(char),cudaMemcpyHostToDevice));

	/*
		Dimensione los bloques de hilos, o utilice las funciones 
		de thrust necesarias.
	*/
	int nk = 1280;
	int nThreads = 64;
	dim3 threads(nThreads);
	dim3 blocks((nk+nThreads-1)/nThreads);
	printf("blocks:%d\n",blocks.x );
	size_t shmem_size = (nThreads)*sizeof(int);
	size_t nonce_size = blocks.x*sizeof(int);

	unsigned int *d_nonce;
	HANDLE_ERROR(cudaMalloc((void**)&d_nonce,nonce_size));

	
	mineblock_shared<<<blocks,threads,shmem_size>>>(d_block,DIFFICULTY,d_nonce);
	checkCUDAError("Error en mineblock");
	cudaDeviceSynchronize();

	unsigned int *pnonce = (unsigned int *)malloc(nonce_size);

	HANDLE_ERROR(cudaMemcpy(pnonce,d_nonce,nonce_size,cudaMemcpyDeviceToHost));	
	//
	//	Last reduction on CPU
	//
	header.nonce = 0;
	for (int i = 0; i < blocks.x; ++i)
	{
		printf("nonce[%d] : %u\n",i,pnonce[i] );
		if(pnonce[i] < DIFFICULTY && pnonce[i]>header.nonce){			
			header.nonce = pnonce[i];
			exit;
			}
	}

	printf("Block header : ");
	hexdump((unsigned char*)&header, sizeof(block_header));
	printf("Nonce = %d\n", header.nonce);
	return(0);


}
