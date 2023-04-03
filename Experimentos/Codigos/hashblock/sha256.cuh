#ifndef CRC_CUH
#define CRC_CUH

/*************************** HEADER FILES ***************************/
#include <stddef.h>

/****************************** MACROS ******************************/
#define HASHSIZE 32            

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte

/*********************** FUNCTION DECLARATIONS **********************/
__device__ crc crcSlow(unsigned char const message[], int nBytes)
__device__ unsigned int sha256silly(const BYTE *buf, size_t buflength);

#endif   // SHA256_CUH
