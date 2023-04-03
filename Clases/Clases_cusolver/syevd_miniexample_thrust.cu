#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <thrust/copy.h>
# include <iostream>


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}


template<typename T>
T* raw(thrust::device_vector<T> &x){
	return thrust::raw_pointer_cast(x.data());
}


int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    const int m = 3;
    const int lda = m;
/*       | 3.5 0.5 0 |
 *   A = | 0.5 3.5 0 |
 *       | 0   0   2 |
 *
 */
    double A[lda*m] = { 3.5, 0.5, 0, 0.5, 3.5, 0, 0, 0, 2.0};
    double lambda[m] = { 2.0, 3.0, 4.0};

    double V[lda*m]; // eigenvectors
    double W[m]; // eigenvalues

    thrust::device_vector<double> d_A(A,A+lda*m);	
    double *d_A_raw = raw(d_A);

    thrust::device_vector<double> d_W(W,W+m);	
    double *d_W_raw = raw(d_W);	

    thrust::device_vector<int> devInfo(1);	
    int *devInfo_raw = raw(devInfo);

    int  lwork = 0;

    int info_gpu = 0;

    printf("A = (matlab base-1)\n");
    printMatrix(m, m, A, lda, "A");
    printf("=====\n");

// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

// step 2: copy A and B to device
    thrust::copy(A, A+lda*m,d_A.begin());

// step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A_raw,
        lda,
        d_W_raw,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    thrust::device_vector<double> d_work(lwork);	
    double *d_work_raw =  raw(d_work);

// step 4: compute spectrum
    cusolver_status = cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A_raw,
        lda,
        d_W_raw,
        d_work_raw,
        lwork,
        devInfo_raw);
    cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    thrust::copy(d_W.begin(),d_W.end(),W);	
    thrust::copy(d_A.begin(), d_A.end(), V);	
    info_gpu=devInfo[0];	

    printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    printf("eigenvalue = (matlab base-1), ascending order\n");
    for(int i = 0 ; i < m ; i++){
        printf("W[%d] = %E\n", i+1, W[i]);
    }

    printf("V = (matlab base-1)\n");
    printMatrix(m, m, V, lda, "V");
    printf("=====\n");

// step 4: check eigenvalues
    double lambda_sup = 0;
    for(int i = 0 ; i < m ; i++){
        double error = fabs( lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error)? lambda_sup : error;
    }
    printf("|lambda - W| = %E\n", lambda_sup);

    if (cusolverH) cusolverDnDestroy(cusolverH);
 
    return 0;
}

