// cutlass_gated_mlp.cu

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>


/// HELPER FUNCTIONS
#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", \
      cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUDA_LAST() \
  CHECK_CUDA(cudaGetLastError())

inline void check_cutlass(
    cutlass::Status status,
    const char* where
) {
    if (status != cutlass::Status::kSuccess) {
        std::cerr
            << "CUTLASS error at " << where << ": "
            << cutlassGetStatusString(status)
            << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
//////////////

#include <cutlass/numeric_types.h>

// --- Input Type Logic ---
#if defined(USE_DOUBLE)
    using Element = double;
#elif defined(USE_FLOAT)
    using Element = float;
#elif defined(USE_HALF)
    using Element = cutlass::half_t;
#elif defined(USE_BF16)
    using Element = cutlass::bfloat16_t;
#elif defined(USE_TF32)
    using Element = cutlass::tfloat32_t;
#elif defined(USE_INT8)
    using Element = int8_t;
#elif defined(USE_UINT8)
    using Element = uint8_t;
#elif defined(USE_E4M3)
    using Element = cutlass::float_e4m3_t;
#elif defined(USE_E5M2)
    using Element = cutlass::float_e5m2_t;
#elif defined(USE_INT4)
    using Element = cutlass::int4b_t;
#else
    #error "Please define a valid input type: USE_DOUBLE, USE_FLOAT, USE_HALF, USE_BF16, USE_TF32, USE_INT8, USE_UINT8, USE_E4M3, USE_E5M2, USE_INT4"
#endif

// --- Accumulator Type Logic ---
#if defined(ACCUM_DOUBLE)
    using Accum = double;
#elif defined(ACCUM_FLOAT)
    using Accum = float;
#elif defined(ACCUM_INT32)
    using Accum = int32_t;
#elif defined(ACCUM_HALF)
    using Accum = cutlass::half_t; // Common in purely half-precision SIMT
#else
    // Fallback logic: Default to reasonable hardware defaults if ACCUM isn't explicit
    #if defined(USE_DOUBLE)
        using Accum = double;
    #elif defined(USE_INT8) || defined(USE_UINT8) || defined(USE_INT4)
        using Accum = int32_t;
    #else
        using Accum = float; // Default for FP16, BF16, TF32, and FP32
    #endif
#endif

//using Element = float;
using Layout  = cutlass::layout::RowMajor;
using LayoutCol = cutlass::layout::ColumnMajor; // Need transposed
//using Accum   = float;

////////////////////////////////////////////////////////////////////////////////
// Hadamard kernel: W = U ∘ V
////////////////////////////////////////////////////////////////////////////////

__global__ void hadamard_kernel(
    const Element* U,
    const Element* V,
    Element* W,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) W[i] = U[i] * V[i];
}

////////////////////////////////////////////////////////////////////////////////
// GEMM types
////////////////////////////////////////////////////////////////////////////////

// V = ReLU(X Bᵀ)
using GemmXB =
cutlass::gemm::device::Gemm<
    Element, Layout,       // X
    Element, LayoutCol,    // B
    Element, Layout,       // V
    Accum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,8>,
    cutlass::epilogue::thread::LinearCombinationRelu<
        Element, 1, Accum, Accum>
>;

// U = X Aᵀ
using GemmXA =
cutlass::gemm::device::Gemm<
    Element, Layout,
    Element, LayoutCol,
    Element, Layout,
    Accum
>;

// Y = W C
using GemmWC =
cutlass::gemm::device::Gemm<
    Element, Layout,
    Element, Layout,
    Element, Layout,
    Accum
>;


void run_mlp(
    Element* X, Element* A, Element* B, Element* C,
    Element* U, Element* V, Element* W, Element* Y,
    int N, int M, int K, int P)
{
    // 1) V = ReLU(X Bᵀ)
    {
        GemmXB gemm;

        check_cutlass(
        gemm({
            {N, M, K},
            {X, K},
            {B, K},
            {V, M},
            {V, M},
            {1.f, 0.f}
        }),
        "GemmXB");
    }

    // 2) U = X Aᵀ
    {
        GemmXA gemm;
        check_cutlass(
        gemm({
            {N, M, K},
            {X, K},
            {A, K},
            {U, M},
            {U, M},
            {1.f, 0.f}
        }),
        "GemmXA");
    }

    // 3) W = U ∘ V
    int threads = 256;
    int blocks  = (N * M + threads - 1) / threads;
    hadamard_kernel<<<blocks, threads>>>(U, V, W, N * M);
    CHECK_CUDA_LAST();

    // 4) Y = W C
    {
        GemmWC gemm;
        check_cutlass(
        gemm({
            {N, P, M},
            {W, M},
            {C, P},
            {Y, P},
            {Y, P},
            {1.f, 0.f}
        }),"GemmWC");
    }
}


////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////

int main() {

    int N = 4096, M = 4096, K = 4096, P = 4096;

    size_t sizeX = N * K * sizeof(Element);
    size_t sizeA = M * K * sizeof(Element);
    size_t sizeB = M * K * sizeof(Element);
    size_t sizeC = M * P * sizeof(Element);
    size_t sizeNM = N * M * sizeof(Element);
    size_t sizeY  = N * P * sizeof(Element);

    // Device buffers
    Element *X, *A, *B, *C, *U, *V, *W, *Y;

    CHECK_CUDA(cudaMalloc(&X, sizeX));
    CHECK_CUDA(cudaMalloc(&A, sizeA));
    CHECK_CUDA(cudaMalloc(&B, sizeB));
    CHECK_CUDA(cudaMalloc(&C, sizeC));
    CHECK_CUDA(cudaMalloc(&U, sizeNM));
    CHECK_CUDA(cudaMalloc(&V, sizeNM));
    CHECK_CUDA(cudaMalloc(&W, sizeNM));
    CHECK_CUDA(cudaMalloc(&Y, sizeY));

    // Optional: initialize with cudaMemset for benchmarking
    CHECK_CUDA(cudaMemset(X, 0, sizeX));
    CHECK_CUDA(cudaMemset(A, 0, sizeA));
    CHECK_CUDA(cudaMemset(B, 0, sizeB));
    CHECK_CUDA(cudaMemset(C, 0, sizeC));

    thrust::fill(thrust::cuda::par, X, X + N * K, 1.0f);
    CHECK_CUDA_LAST();

    thrust::fill(thrust::cuda::par, A, A + M * K, 1.0f);
    CHECK_CUDA_LAST();

    thrust::fill(thrust::cuda::par, B, B + M * K, 1.0f);
    CHECK_CUDA_LAST();

    thrust::fill(thrust::cuda::par, C, C + M * P, 1.0f);
    CHECK_CUDA_LAST();

    // warming up
    for(int i = 0; i < 10; i++)
    run_mlp(X, A, B, C, U, V, W, Y, N, M, K, P);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < 10; i++) {
        run_mlp(X, A, B, C, U, V, W, Y, N, M, K, P);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUTLASS gated MLP time: %.3f ms\n", ms/10);

    cudaDeviceSynchronize();

    // print some outputs
    thrust::host_vector<Element> hY(N * P);
    CHECK_CUDA_LAST();

    CHECK_CUDA(cudaMemcpy(hY.data(), Y, sizeY, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) printf("Y[%d] = %f\n", i, float(hY[i]));

    // Cleanup
    cudaFree(X); cudaFree(A); cudaFree(B); cudaFree(C);
    cudaFree(U); cudaFree(V); cudaFree(W); cudaFree(Y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Done.\n");
    return 0;
}
