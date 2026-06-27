#include "ff/precision.h"

namespace tinker {
// template <class T>
// void zeroOnDevice3Async_acc(int nelem, T* a1, T* a2, T* a3)
// {
//    #pragma acc parallel loop async deviceptr(a1,a2,a3)
//    for (int i = 0; i < nelem; ++i) {
//       if (a1)
//          a1[i] = 0;
//       if (a2)
//          a2[i] = 0;
//       if (a3)
//          a3[i] = 0;
//    }
// }
// template void zeroOnDevice3Async_acc(int, fixed*, fixed*, fixed*);
// template void zeroOnDevice3Async_acc(int, float*, float*, float*);
// template void zeroOnDevice3Async_acc(int, double*, double*, double*);

template <class T>
void zeroOnDevice4Async_acc(int nelem, T* a1, T* a2, T* a3, T* a4)
{
   #pragma acc parallel loop async deviceptr(a1,a2,a3,a4)
   for (int i = 0; i < nelem; ++i) {
      if (a1)
         a1[i] = 0;
      if (a2)
         a2[i] = 0;
      if (a3)
         a3[i] = 0;
      if (a4)
         a4[i] = 0;
   }
}
template void zeroOnDevice4Async_acc(int, fixed*, fixed*, fixed*, fixed*);
template void zeroOnDevice4Async_acc(int, float*, float*, float*, float*);
template void zeroOnDevice4Async_acc(int, double*, double*, double*, double*);

// template <class T, int N>
// void zeroOnDevice3Async_acc(int nelem, T (*b1)[N], T (*b2)[N], T (*b3)[N])
// {
//    T* a1 = (T*)b1;
//    T* a2 = (T*)b2;
//    T* a3 = (T*)b3;
//    #pragma acc parallel loop async deviceptr(a1,a2,a3)
//    for (int i = 0; i < N * nelem; ++i) {
//       if (a1)
//          a1[i] = 0;
//       if (a2)
//          a2[i] = 0;
//       if (a3)
//          a3[i] = 0;
//    }
// }
// template void zeroOnDevice3Async_acc(int, fixed (*)[8], fixed (*)[8], fixed (*)[8]);
// template void zeroOnDevice3Async_acc(int, float (*)[8], float (*)[8], float (*)[8]);
// template void zeroOnDevice3Async_acc(int, double (*)[8], double (*)[8], double (*)[8]);

template <class T, int N>
void zeroOnDevice4Async_acc(int nelem, T (*b1)[N], T (*b2)[N], T (*b3)[N], T (*b4)[N])
{
   T* a1 = (T*)b1;
   T* a2 = (T*)b2;
   T* a3 = (T*)b3;
   T* a4 = (T*)b4;
   #pragma acc parallel loop async deviceptr(a1,a2,a3,a4)
   for (int i = 0; i < N * nelem; ++i) {
      if (a1)
         a1[i] = 0;
      if (a2)
         a2[i] = 0;
      if (a3)
         a3[i] = 0;
      if (a4)
         a4[i] = 0;
   }
}
template void zeroOnDevice4Async_acc(int, fixed (*)[8], fixed (*)[8], fixed (*)[8], fixed (*)[8]);
template void zeroOnDevice4Async_acc(int, float (*)[8], float (*)[8], float (*)[8], float (*)[8]);
template void zeroOnDevice4Async_acc(int, double (*)[8], double (*)[8], double (*)[8], double (*)[8]);

// template <class T>
// void zeroOnDevice9Async_acc(
//    int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9)
// {
//    #pragma acc parallel loop async deviceptr(a1,a2,a3,a4,a5,a6,a7,a8,a9)
//    for (int i = 0; i < nelem; ++i) {
//       if (a1)
//          a1[i] = 0;
//       if (a2)
//          a2[i] = 0;
//       if (a3)
//          a3[i] = 0;
//       if (a4)
//          a4[i] = 0;
//       if (a5)
//          a5[i] = 0;
//       if (a6)
//          a6[i] = 0;
//       if (a7)
//          a7[i] = 0;
//       if (a8)
//          a8[i] = 0;
//       if (a9)
//          a9[i] = 0;
//    }
// }
// template void zeroOnDevice9Async_acc(
//    int, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*);
// template void zeroOnDevice9Async_acc(
//    int, float*, float*, float*, float*, float*, float*, float*, float*, float*);
// template void zeroOnDevice9Async_acc(
//    int, double*, double*, double*, double*, double*, double*, double*, double*, double*);

template <class T>
void zeroOnDevice12Async_acc(int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9, T* a10, T* a11,
   T* a12)
{
   #pragma acc parallel loop async deviceptr(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)
   for (int i = 0; i < nelem; ++i) {
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
      if (a10)
         a10[i] = 0;
      if (a11)
         a11[i] = 0;
      if (a12)
         a12[i] = 0;
   }
}
template void zeroOnDevice12Async_acc(int, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*, fixed*,
   fixed*, fixed*, fixed*);
template void zeroOnDevice12Async_acc(int, float*, float*, float*, float*, float*, float*, float*, float*, float*,
   float*, float*, float*);
template void zeroOnDevice12Async_acc(int, double*, double*, double*, double*, double*, double*, double*, double*,
   double*, double*, double*, double*);
}
