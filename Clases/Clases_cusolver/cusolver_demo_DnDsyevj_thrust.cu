# include <time.h>
# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include <cusolverDn.h>
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/copy.h>
# include <iostream>

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
	int m = 2048;
	if(argc!=2) {
		std::cout << "uso:" << argv[0] << " nfilas" << std::endl;
		std::cout << "continuamos con default, nfilas=2048" << std::endl;
	}
	else m = atoi(argv[1]);

	// leading dimension of A
	const int lda = m ;

	// mxm matrix
	thrust::host_vector<float> A(lda*m);
	float * A_raw = thrust::raw_pointer_cast(A.data());

	// mxm matrix of eigenvectors
	thrust::host_vector<float> V(lda*m);
	float * V_raw = thrust::raw_pointer_cast(V.data()) ;

	// m - vector of eigenvalues
	thrust::host_vector<float> W(m);
	float * W_raw=thrust::raw_pointer_cast(W.data())  ;
	
	// define random A
	for ( int i =0; i < A.size() ; i ++) A[i] = rand()/(float)RAND_MAX;

	// declare arrays on the device

	// mxm matrix A on the device
	thrust::device_vector<float> d_A(A);
	float *d_A_raw=thrust::raw_pointer_cast(d_A.data());

	// m - vector of eigenvalues on the device
	thrust::device_vector<float> d_W(m);
	float *d_W_raw=thrust::raw_pointer_cast(d_W.data());

	// info on the device
	thrust::device_vector<int> devInfo(1);
	int * devInfo_raw=thrust::raw_pointer_cast(devInfo.data());

	// workspace size
	int lwork = 0;

	// info copied from device to host
	int info_gpu = 0;

	// create cusolver handle
	cusolver_status = cusolverDnCreate(&cusolverH);

	// copy A - > d_A
	thrust::copy(A.begin(),A.end(),d_A.begin());	
	
	// compute eigenvalues and eigenvectors
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR ;
	
	// use lower left triangle of the matrix
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;
	
	// compute buffer size and prepare workspace
	cusolver_status = cusolverDnSsyevd_bufferSize ( cusolverH ,
	jobz , uplo , m , d_A_raw , lda , d_W_raw , &lwork );	

	// workspace on the device
	thrust::device_vector<float> d_work(lwork);
	float * d_work_raw=thrust::raw_pointer_cast(d_work.data());
	
	// start timer
	clock_gettime ( CLOCK_REALTIME ,&start );
	
	// compute the eigenvalues and eigenvectors for a symmetric ,
	// real mxm matrix ( only the lower left triangle af A is used )
	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m,
	d_A_raw, lda, d_W_raw, d_work_raw, lwork, devInfo_raw);


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
	thrust::copy(d_W.begin(),d_W.end(),W.begin());

	// copy d_A - > V
	thrust::copy(d_A.begin(),d_A.end(),V.begin());	

	 // copy devInfo - > info_gpu
	info_gpu=devInfo[0];
	printf ( " after syevd : info_gpu = %d \n " , info_gpu );

	// print first eigenvalues
	printf ( "primeros 3 eigenvalues :\n " );
	for ( int i = 0 ; i < 3 ; i ++){
	printf ( " W[%d] = %E \n " , i +1 , W_raw[i]);
	}

	cusolver_status = cusolverDnDestroy( cusolverH );
	if(cusolver_status != CUSOLVER_STATUS_SUCCESS ){std::cout << "error destroying context" << std::endl;};

	return 0;
}
