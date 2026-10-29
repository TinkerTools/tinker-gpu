#include "seq/launch.h"

namespace tinker {
template <class T>
__global__
void zeroOnDeviceAsync_cu1(int n, T* restrict a1)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (a1)
         a1[i] = 0;
   }
}

template <class T>
void zeroOnDeviceAsync_cu(int nelem, T* a1)
{
   launch_k1s(g::s0, nelem, zeroOnDeviceAsync_cu1<T>, nelem, a1);
}
template void zeroOnDeviceAsync_cu(int, fixed*);
template void zeroOnDeviceAsync_cu(int, float*);
template void zeroOnDeviceAsync_cu(int, double*);

template <class T, int N>
__global__
void zeroOnDeviceAsync_cu2(int n, T (*restrict b1)[N])
{
   T* restrict a1 = reinterpret_cast<T*>(b1);
   for (int i = ITHREAD; i < n * N; i += STRIDE) {
      if (a1)
         a1[i] = 0;
   }
}

template <class T, int N>
void zeroOnDeviceAsync_cu(int nelem, T (*a1)[N])
{
   launch_k1s(g::s0, nelem, zeroOnDeviceAsync_cu2<T, N>, nelem, a1);
}
template void zeroOnDeviceAsync_cu(int, fixed (*)[8]);
template void zeroOnDeviceAsync_cu(int, float (*)[8]);
template void zeroOnDeviceAsync_cu(int, double (*)[8]);

template <class T>
__global__
void zeroOnDevice3Async_cu1(int n, T* restrict a1, T* restrict a2, T* restrict a3)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (a1)
         a1[i] = 0;
      if (a2)
         a2[i] = 0;
      if (a3)
         a3[i] = 0;
   }
}

template <class T>
void zeroOnDevice3Async_cu(int nelem, T* a1, T* a2, T* a3)
{
   launch_k1s(g::s0, nelem, zeroOnDevice3Async_cu1<T>, nelem, a1, a2, a3);
}
template void zeroOnDevice3Async_cu(int, fixed*, fixed*, fixed*);
template void zeroOnDevice3Async_cu(int, float*, float*, float*);
template void zeroOnDevice3Async_cu(int, double*, double*, double*);

template <class T, int N>
__global__
void zeroOnDevice3Async_cu2(int n, T (*restrict b1)[N], T (*restrict b2)[N], T (*restrict b3)[N])
{
   T* restrict a1 = reinterpret_cast<T*>(b1);
   T* restrict a2 = reinterpret_cast<T*>(b2);
   T* restrict a3 = reinterpret_cast<T*>(b3);
   for (int i = ITHREAD; i < n * N; i += STRIDE) {
      if (a1)
         a1[i] = 0;
      if (a2)
         a2[i] = 0;
      if (a3)
         a3[i] = 0;
   }
}

template <class T, int N>
void zeroOnDevice3Async_cu(int nelem, T (*a1)[N], T (*a2)[N], T (*a3)[N])
{
   launch_k1s(g::s0, nelem, zeroOnDevice3Async_cu2<T, N>, nelem, a1, a2, a3);
}
template void zeroOnDevice3Async_cu(int, fixed (*)[8], fixed (*)[8], fixed (*)[8]);
template void zeroOnDevice3Async_cu(int, float (*)[8], float (*)[8], float (*)[8]);
template void zeroOnDevice3Async_cu(int, double (*)[8], double (*)[8], double (*)[8]);

template <class T>
__global__
void zeroOnDevice9Async_cu1(int n, T* restrict a1, T* restrict a2, T* restrict a3, T* restrict a4,
   T* restrict a5, T* restrict a6, T* restrict a7, T* restrict a8, T* restrict a9)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (a1)
         a1[i] = 0;
      if (a2)
         a2[i] = 0;
      if (a3)
         a3[i] = 0;
      if (a4)
         a4[i] = 0;
      if (a5)
         a5[i] = 0;
      if (a6)
         a6[i] = 0;
      if (a7)
         a7[i] = 0;
      if (a8)
         a8[i] = 0;
      if (a9)
         a9[i] = 0;
   }
}

template <class T>
void zeroOnDevice9Async_cu(int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9)
{
   launch_k1s(g::s0, nelem, zeroOnDevice9Async_cu1<T>, nelem, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
template void zeroOnDevice9Async_cu(
   int, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*);
template void zeroOnDevice9Async_cu(
   int, float*, float*, float*, float*, float*, float*, float*, float*, float*);
template void zeroOnDevice9Async_cu(
   int, double*, double*, double*, double*, double*, double*, double*, double*, double*);
}
