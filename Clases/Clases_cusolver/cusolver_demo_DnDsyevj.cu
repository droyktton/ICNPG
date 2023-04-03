# include <time.h>
# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include <cusolverDn.h>
# define BILLION 1000000000L;


int main ( int argc , char* argv[])
{
	struct timespec start , stop ;

	// variables for timing
	double accum;

	// elapsed time variable
	cusolverDnHandle_t cusolverH;
	cudaError_t cudaStat = cudaSuccess ;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS ;

	// number of rows and columns of A
	const int m = 2048;

	// leading dimension of A
	const int lda = m ;

	// mxm matrix
	float * A ;

	// mxm matrix of eigenvectors
	float * V ;

	// m - vector of eigenvalues
	float * W ;

	// prepare memory on the host
	A = ( float *) malloc ( lda * m * sizeof ( float ));
	V = ( float *) malloc ( lda * m * sizeof ( float ));
	W = ( float *) malloc ( m * sizeof ( float ));

	// define random A
	for ( int i =0; i < lda * m ; i ++) A [ i ] = rand ()/( float ) RAND_MAX ;

	// declare arrays on the device

	// mxm matrix A on the device
	float * d_A ;

	// m - vector of eigenvalues on the device
	float * d_W ;

	// info on the device
	int * devInfo ;

	// workspace on the device
	float * d_work ;

	// workspace size
	int lwork = 0;

	// info copied from device to host
	int info_gpu = 0;

	// create cusolver handle
	cusolver_status = cusolverDnCreate(&cusolverH);

	// prepare memory on the device
	cudaStat = cudaMalloc (( void **)&d_A , sizeof ( float )* lda * m );
	cudaStat = cudaMalloc (( void **)&d_W , sizeof ( float )* m );
	cudaStat = cudaMalloc (( void **)&devInfo , sizeof ( int ));

	// copy A - > d_A
	cudaStat = cudaMemcpy ( d_A ,A , sizeof ( float )*lda*m ,
	cudaMemcpyHostToDevice);
	
	// compute eigenvalues and eigenvectors
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR ;
	
	// use lower left triangle of the matrix
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;
	
	// compute buffer size and prepare workspace
	cusolver_status = cusolverDnSsyevd_bufferSize ( cusolverH ,
	jobz , uplo , m , d_A , lda , d_W , &lwork );	
	cudaStat = cudaMalloc (( void **)&d_work , sizeof ( float )* lwork );
	
	// start timer
	clock_gettime ( CLOCK_REALTIME ,&start );
	
	// compute the eigenvalues and eigenvectors for a symmetric ,
	// real mxm matrix ( only the lower left triangle af A is used )
	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m,
	d_A, lda, d_W, d_work, lwork, devInfo);

	if(cusolver_status!=CUSOLVER_STATUS_SUCCESS){/*avisar!*/};

	cudaStat = cudaDeviceSynchronize();
	if(cudaStat!=cudaSuccess){/*avisar!*/};
 
	// stop timer
	clock_gettime ( CLOCK_REALTIME ,&stop );

	// elapsed time
	accum=(stop.tv_sec-start.tv_sec)+(stop.tv_nsec-start.tv_nsec)/(double)BILLION;

	// print elapsed time
	printf( " Ssyevd time : %lf sec .\n " , accum ); 

	// copy d_W - > W
	cudaStat = cudaMemcpy (W , d_W , sizeof ( float )* m ,
	cudaMemcpyDeviceToHost );

	// copy d_A - > V
	cudaStat = cudaMemcpy (V , d_A , sizeof ( float )* lda *m ,
	cudaMemcpyDeviceToHost );

	 // copy devInfo - > info_gpu
	cudaStat = cudaMemcpy (& info_gpu , devInfo , sizeof ( int ) ,
	cudaMemcpyDeviceToHost );
	printf ( " after syevd : info_gpu = %d \n " , info_gpu );

	// print first eigenvalues
	printf ( "primeros 3 eigenvalues :\n " );
	for ( int i = 0 ; i < 3 ; i ++){
	printf ( " W [%d ] = %E \n " , i +1 , W [ i ]);
	}

	// free memory
	cudaFree( d_A );
	cudaFree( d_W );
	cudaFree( devInfo );
	cudaFree( d_work );
	cusolverDnDestroy( cusolverH );
	cudaDeviceReset();
	return 0;
}
