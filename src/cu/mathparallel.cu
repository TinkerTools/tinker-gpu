#include "math/parallelcu.h"
#include "seq/reduce.h"
#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/error.h"
#include "tool/gpucard.h"
#include "tool/ptrtrait.h"
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <numeric>

namespace tinker {
inline namespace v1 {
template <class T, class Op>
void reduce_to_dptr(const T* a, size_t nelem, cudaStream_t st)
{
   T* dptr = (T*)dptr_buf;
   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   int grid_siz2 = (nelem + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   reduce<T, BLOCK_DIM, Op><<<grid_size, BLOCK_DIM, 0, st>>>(dptr, a, nelem);
   reduce<T, BLOCK_DIM, Op><<<1, BLOCK_DIM, 0, st>>>(dptr, dptr, grid_size);
}

template <class T, class Op>
T reduce_general(const T* a, size_t nelem, int queue)
{
   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
   T* dptr = (T*)dptr_buf;
   T* hptr = (T*)pinned_buf;
   reduce_to_dptr<T, Op>(a, nelem, st);
   check_rt(cudaMemcpyAsync(hptr, dptr, sizeof(T), cudaMemcpyDeviceToHost, st));
   // always wait
   check_rt(cudaStreamSynchronize(st));
   return *hptr;
}
}

template <class T>
T reduceSum_cu(const T* a, size_t nelem, int queue)
{
   return reduce_general<T, OpPlus<T>>(a, nelem, queue);
}
template int reduceSum_cu(const int*, size_t, int);
template float reduceSum_cu(const float*, size_t, int);
template double reduceSum_cu(const double*, size_t, int);
template unsigned long long reduceSum_cu(const unsigned long long*, size_t, int);

template <class HT, size_t HN, class DPTR>
void reduceSum2_cu(HT (&restrict h_ans)[HN], DPTR restrict a, size_t nelem, int queue)
{
   typedef typename PtrTrait<DPTR>::type CONST_DT;
   typedef typename std::remove_const<CONST_DT>::type T;
   static_assert(std::is_same<HT, T>::value, "");
   constexpr size_t N = PtrTrait<DPTR>::n;
   static_assert(HN <= N, "");

   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
   T(*dptr)[HN] = (T(*)[HN])dptr_buf;
   T* hptr = (T*)pinned_buf;
   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   grid_siz1 = grid_siz1 / HN; // limited by the output buffer
   int grid_siz2 = (nelem + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   reduce2<T, BLOCK_DIM, HN, N, OpPlus<T>><<<grid_size, BLOCK_DIM, 0, st>>>(dptr, a, nelem);
   reduce2<T, BLOCK_DIM, HN, HN, OpPlus<T>><<<1, BLOCK_DIM, 0, st>>>(dptr, dptr, grid_size);
   check_rt(cudaMemcpyAsync(hptr, (T*)dptr, HN * sizeof(HT), cudaMemcpyDeviceToHost, st));
   // always wait
   check_rt(cudaStreamSynchronize(st));
   #pragma unroll
   for (size_t j = 0; j < HN; ++j)
      h_ans[j] = hptr[j];
}
template void reduceSum2_cu(float (&)[6], float (*)[8], size_t, int);
template void reduceSum2_cu(double (&)[6], double (*)[8], size_t, int);
template void reduceSum2_cu(unsigned long long (&)[6], unsigned long long (*)[8], size_t, int);

template <class T>
void reduceSumOnDevice_cu(T* dp_ans, const T* a, size_t nelem, int queue)
{
   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
   T* dptr = (T*)dptr_buf;
   using Op = OpPlus<T>;

   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   int grid_siz2 = (nelem + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   reduce<T, BLOCK_DIM, Op><<<grid_size, BLOCK_DIM, 0, st>>>(dptr, a, nelem);
   reduce<T, BLOCK_DIM, Op><<<1, BLOCK_DIM, 0, st>>>(dp_ans, dptr, grid_size);
}
template void reduceSumOnDevice_cu(int*, const int*, size_t, int);
template void reduceSumOnDevice_cu(float*, const float*, size_t, int);
template void reduceSumOnDevice_cu(double*, const double*, size_t, int);
template void reduceSumOnDevice_cu(unsigned long long*, const unsigned long long*, size_t, int);

template <class HT, size_t HN, class DPTR>
void reduceSum2OnDevice_cu(HT (&dref)[HN], DPTR v, size_t nelem, int queue)
{
   typedef typename PtrTrait<DPTR>::type CONST_DT;
   typedef typename std::remove_const<CONST_DT>::type T;
   static_assert(std::is_same<HT, T>::value, "");
   constexpr size_t N = PtrTrait<DPTR>::n;
   static_assert(HN <= N, "");

   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
   T(*dptr)[HN] = (T(*)[HN])dptr_buf;
   T(*dpt2)[HN] = (T(*)[HN])dref;
   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   grid_siz1 = grid_siz1 / HN; // limited by the output buffer
   int grid_siz2 = (nelem + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   reduce2<T, BLOCK_DIM, HN, N, OpPlus<T>><<<grid_size, BLOCK_DIM, 0, st>>>(dptr, v, nelem);
   reduce2<T, BLOCK_DIM, HN, HN, OpPlus<T>><<<1, BLOCK_DIM, 0, st>>>(dpt2, dptr, grid_size);
}
template void reduceSum2OnDevice_cu(float (&)[6], float (*)[8], size_t, int);
template void reduceSum2OnDevice_cu(double (&)[6], double (*)[8], size_t, int);
template void reduceSum2OnDevice_cu(unsigned long long (&)[6], unsigned long long (*)[8], size_t, int);

template <>
void dotProd_cu<float>(float* ans, const float* a, const float* b, size_t nelem, int queue)
{
   bool dq = queue == g::q1;
   cublasHandle_t hd = (dq ? g::h1 : g::h0);
   check_rt(cublasSdot(hd, nelem, a, 1, b, 1, ans));
}

template <>
void dotProd_cu<double>(double* ans, const double* a, const double* b, size_t nelem, int queue)
{
   bool dq = queue == g::q1;
   cublasHandle_t hd = (dq ? g::h1 : g::h0);
   check_rt(cublasDdot(hd, nelem, a, 1, b, 1, ans));
}

// cublas gemm does not run as fast here prior to cuda 10.1.
// Old code:
//
// #if CUDART_VERSION >= 10100 // >= 10.1
//    float alpha = 1, beta = 0;
//    check_rt(cublasSgemm(hd, CUBLAS_OP_N, CUBLAS_OP_T, 1, 1, nelem, //
//                         &alpha, a, 1, b, 1,                        //
//                         &beta, ans, 1));
// #else
//    check_rt(cublasSdot(hd, nelem, a, 1, b, 1, ans));
// #endif

template <class T>
__global__
void scaleArray_cu1(T* dst, T scal, size_t nelem)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nelem; i += blockDim.x * gridDim.x) {
      dst[i] *= scal;
   }
}

template <class T>
void scaleArray_cu(T* dst, T scal, size_t nelem, int queue)
{
   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
   int gs = (nelem + BLOCK_DIM - 1) / BLOCK_DIM;
   auto ker = scaleArray_cu1<T>;
   ker<<<gs, BLOCK_DIM, 0, st>>>(dst, scal, nelem);
}
template void scaleArray_cu(float*, float, size_t, int);
template void scaleArray_cu(double*, double, size_t, int);

template <>
void genMatMul_cu<float>(float* c, const float* a, const float* b, int m, int n, int k, bool transa, bool transb,
   const float* alpha, const float* beta, int queue)
{
   bool dq = queue == g::q1;
   cublasHandle_t hd = (dq ? g::h1 : g::h0);
   cublasOperation_t opa = (transa ? CUBLAS_OP_T : CUBLAS_OP_N);
   cublasOperation_t opb = (transb ? CUBLAS_OP_T : CUBLAS_OP_N);
   check_rt(cublasSgemm(hd, opa, opb, m, n, k, //
      alpha, a, (transa ? k : m), b, (transb ? n : k), beta, c, m));
}

template <>
void genMatMul_cu<double>(double* c, const double* a, const double* b, int m, int n, int k, bool transa, bool transb,
   const double* alpha, const double* beta, int queue)
{
   bool dq = queue == g::q1;
   cublasHandle_t hd = (dq ? g::h1 : g::h0);
   cublasOperation_t opa = (transa ? CUBLAS_OP_T : CUBLAS_OP_N);
   cublasOperation_t opb = (transb ? CUBLAS_OP_T : CUBLAS_OP_N);
   check_rt(cublasDgemm(hd, opa, opb, m, n, k, alpha, a, (transa ? k : m), b, (transb ? n : k), beta, c, m));
}

}
