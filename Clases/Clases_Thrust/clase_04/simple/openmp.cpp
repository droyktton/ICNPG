#include <stdio.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include "cpu_timer.h"
#include <iostream>

using namespace thrust::placeholders;

struct suma
{
    __host__ __device__
    int operator()(int x, int y) const
    {
	int c=x+y;
	#ifdef PRINT
	int i=omp_get_thread_num();
        printf("thrust omp thread %d calculates c[%d] = %d\n", i, i, c);
	#endif
	return c;
    }
};


int main(int argc, char**argv) 
{
    int i, n = atoi(argv[1]);


    //int a[n], b[n], c[n];
    int *a=(int *)malloc(sizeof(int)*n);
    int *b=(int *)malloc(sizeof(int)*n);
    int *c=(int *)malloc(sizeof(int)*n);

    // Initialize arrays
    for (i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cpu_timer reloj;


    //double start_time = omp_get_wtime();
    reloj.tic();
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i];

	#ifdef PRINT
        printf("omp thread %d calculates c[%d] = %d\n", omp_get_thread_num(), i, c[i]);
	#endif	
    }
    // code to time goes here

    //double end_time = omp_get_wtime();
    //double duration = end_time - start_time;

    double tomp2=reloj.tac();
    printf("omp time = %f ms\n",tomp2);



    thrust::device_vector<int> da(a,a+n);
    thrust::device_vector<int> db(b,b+n);
    thrust::device_vector<int> dc(n,0);


    int max=omp_get_max_threads();
    printf("maximum number of omp threads=%d\n",max);
    for(i=max;i<max+1;i++)
    {
	omp_set_num_threads(i);
    	reloj.tic();
    	thrust::transform(da.begin(),da.end(),db.begin(),dc.begin(),suma());
    	printf("nro de threads=%d, t=%f ms\n",i,reloj.tac());
    }
    return 0;
}

//g++ -fopenmp -lgomp openmp.cpp  -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/targets/x86_64-linux/include/ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
