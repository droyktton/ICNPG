#include<stdio.h>
#include<stdlib.h>

#ifndef SIZE
#define SIZE	2
#endif

#ifdef PRINT
void print_matrix(float *);
#endif

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

  for (i = 0; i < SIZE; ++i) 
  {
	for (j = 0; j < SIZE; ++j) 
	{
      		for (k = 0; k < SIZE; ++k) {
    		c[i*SIZE+j] += a[i*SIZE+k] * b[k*SIZE+j];
	        }
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
			printf("%f ", m[i*SIZE+j]);	
		}
		printf("\n");
	}
}
#endif
