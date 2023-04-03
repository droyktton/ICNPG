#include<stdio.h>
#include<stdlib.h>

#ifndef SIZE
#define SIZE	2
#endif

#ifdef PRINT
void print_matrix(float *);
#endif

#include <cublas_v2.h>

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


int main()
{
  int i,j,k;

  printf("la matriz es de tamanio %d x %d\n", SIZE, SIZE);
  printf("si quiere cambiarlo a xxxx: gcc -DSIZE=xxxx .... \n");
  printf("si quiere imprimir las matrices: gcc -DPRINT .... \n");

  float * restrict a=malloc(sizeof(float)*SIZE*SIZE) ;
  float * restrict b=malloc(sizeof(float)*SIZE*SIZE);
  float * restrict c=malloc(sizeof(float)*SIZE*SIZE);

  // Initialize matrices.
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i*SIZE+j] = (float)i + j;
      b[i*SIZE+j] = (float)i - j;
      c[i*SIZE+j] = 0.0f;
    }
  }

  #pragma acc data copyin(a[0:SIZE*SIZE]) copyin(b[0:SIZE*SIZE]) copyout(c[0:SIZE*SIZE])
  {
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
  }

  // solo para test...	
  #ifdef PRINT	
  printf("a:\n");print_matrix(a);
  printf("b:\n");print_matrix(b);
  printf("c:\n");print_matrix(c);
  #endif

  return 0;
}


#ifdef PRINT
void print_matrix(float *m){
	int i,j;
	for(i=0;i<SIZE;i++){
		for(j=0;j<SIZE;j++){
			printf("%f ", m[j*SIZE+i]);	
		}
		printf("\n");
	}
}
#endif
