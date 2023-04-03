#include<stdio.h>

#ifdef CUBLAS
#include <cublas_v2.h>
#endif

#ifndef SIZE
#define SIZE	2
#endif

#ifdef PRINT
void print_matrix(float [SIZE][SIZE]);
#endif

#ifdef CUBLAS
extern 
cublasStatus_t cublasSgemm(cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda,
                                                      const float *B,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
#endif

float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];
int main()
{
	  int i,j,k;

	  printf("la matriz es de tamanio %d x %d\n", SIZE, SIZE);
	  printf("si quiere cambiarlo a xxxx: gcc -DSIZE=xxxx .... \n");
	  printf("si quiere imprimir las matrices: gcc -DPRINT .... \n");

	  // Initialize matrices.
	  for (i = 0; i < SIZE; ++i) {
	    for (j = 0; j < SIZE; ++j) {
	      a[i][j] = (float)i + j;
	      b[i][j] = (float)i - j;
	      c[i][j] = 0.0f;
	    }
	  }

	#pragma acc data create(a[0:SIZE]) create(b[0:SIZE]) copyout(c[0:SIZE])
	{

	#ifndef CUBLAS
	  	#pragma acc kernels
	  	for (i = 0; i < SIZE; ++i) 
	  	{
			for (j = 0; j < SIZE; ++j) 
			{
	      			for (k = 0; k < SIZE; ++k) {
	    			c[i][j] += a[i][k] * b[k][j];
				}
			}
	  	}
	  	printf("openacc\n");	

	#else

	  	#pragma acc host_data use_device(a,b,c)
	  	{
		    	cublasStatus_t stat;

			const float  al=1.0f;                 
			const float bet =0.0f;
			int m=SIZE;

			cublasHandle_t manija;
			stat=cublasCreate(&manija);
			stat=cublasSgemm(manija,0,0,m,m,m,&al, a,m,b,m, &bet,c,m);
	  	}

	  	printf("cublasSgemm\n");	
	#endif

	}// device a,b,c region

	#ifdef PRINT
	printf("a:\n");print_matrix(a);
	printf("b:\n");print_matrix(b);
	printf("c:\n");print_matrix(c);
	#endif

	  return 0;
}


#ifdef PRINT
void print_matrix(float m[SIZE][SIZE]){
	int i,j;
	for(i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
			printf("%f ", m[i][j]);	
		}
		printf("\n");
	}
}
#endif
