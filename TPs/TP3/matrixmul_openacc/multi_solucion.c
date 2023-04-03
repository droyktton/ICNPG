#include<stdio.h>

#ifndef SIZE
#define SIZE	2
#endif

#ifdef PRINT
void print_matrix(float [SIZE][SIZE]);
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

  // solo para test...	
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
