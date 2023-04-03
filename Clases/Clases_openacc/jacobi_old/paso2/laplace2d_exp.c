/*

Es el ejemplo de Jacobi pero usando arrays unidimensionales

 */

#include<omp.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <stdio.h>
#include <openacc.h>

int main(int argc, char* argv[])
{
    int n = 4096;
    int m = 4096;
    int nm=n*m;

    int iter_max = 1000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float error     = 1.0f;
    
    float * A=(float *)malloc(sizeof(float)*nm);
    float * Anew=(float *)malloc(sizeof(float)*nm);
    float * y0=(float *)malloc(sizeof(float)*n);
 
    memset(A, 0, n * m * sizeof(float));
    
    // set boundary conditions
    for (int i = 0; i < m; i++)
    {
        A[0*n+i]   = 0.f;
        A[(n-1)*n+i] = 0.f;
    }
    
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
        A[j*n+0] = y0[j];
        A[j*n+m-1] = y0[j]*expf(-pi);
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    #if _OPENACC
    acc_init(acc_device_nvidia);
    #endif

    StartTimer();
    int iter = 0;

    #pragma omp parallel for shared(Anew)
    for (int i = 1; i < m; i++)
    {
       Anew[0*n+i]   = 0.f;
       Anew[(n-1)*n+i] = 0.f;
    }

#pragma omp parallel shared(Anew)
{ 
    #pragma omp parallel for 
    for (int j = 1; j < n; j++)
    {
        Anew[j*n+0]   = y0[j];
        Anew[j*n+m-1] = y0[j]*expf(-pi);
    }
    int tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);
}    


    /// loop de convergencia - no paralelizable pero con localidad de los datos
    #pragma acc data copy(A[0:nm]) copyin(Anew[0:nm])
    #pragma omp parallel shared(m, n, Anew, A)
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

	#pragma acc parallel loop collapse(2) reduction(max:error) 
        #pragma omp parallel for reduction(max:error) 
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                Anew[j*n+i] = 0.25f * ( A[j*n+i+1] + A[j*n+i-1]
                                     + A[(j-1)*n+i] + A[(j+1)*n+i]);
                error = fmaxf( error, fabsf(Anew[j*n+i]-A[j*n+i]));
            }
        }

       #pragma acc parallel loop collapse(2) // Alt2
       #pragma omp parallel for 
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j*n+i] = Anew[j*n+i];    
            }
        }

/*	#pragma acc parallel host_data use_device(A,Anew) 
	{
		float *tmp=A;
		A=Anew; Anew=tmp;		
	}
*/	
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
    /// end loop de convergencia

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000.f);

}
