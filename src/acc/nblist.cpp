#include "ff/atom.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "tool/gpucard.h"

#if TINKER_CUDART
namespace tinker {
#define m_swap(a, b) \
   {                 \
      auto tmp = b;  \
      b = a;         \
      a = tmp;       \
   }
#define m_max_heap(arr, start, end)                     \
   {                                                    \
      int dad = start;                                  \
      int son = dad * 2 + 1;                            \
      while (son <= end) {                              \
         if (son + 1 <= end && arr[son] < arr[son + 1]) \
            ++son;                                      \
         if (arr[dad] > arr[son])                       \
            return;                                     \
         else {                                         \
            m_swap(arr[dad], arr[son]);                 \
            dad = son;                                  \
            son = dad * 2 + 1;                          \
         }                                              \
      }                                                 \
   }

#pragma acc routine seq
inline static void nblistSort_acc1(int* arr, int len)
{
   // heapsort
   for (int i = len / 2 - 1; i >= 0; --i) {
      m_max_heap(arr, i, len - 1);
   }

   for (int i = len - 1; i > 0; --i) {
      m_swap(arr[0], arr[i]);
      m_max_heap(arr, 0, i - 1);
   }
}
#undef m_swap
#undef m_max_heap
}
#else
#include <algorithm>
namespace tinker {
inline static void nblistSort_acc1(int* arr, int len)
{
   std::sort(arr, arr + len);
}
}
#endif

namespace tinker {
// double loop
static void nblistBuildDoubleLoop_acc(NBListUnit nu)
{
   auto* lst = nu.deviceptr();
   #pragma acc parallel loop independent async deviceptr(lst)
   for (int i = 0; i < n; ++i) {
      lst->nlst[i] = n - i - 1;
      lst->lst[i] = i + 1;
   }
}

// static void nblistUpdateDoubleLoop_acc() {}

// version 1
// see also nblist.f
static void nblistBuild_acc1(NBListUnit nu)
{
   auto& st = *nu;
   const int maxnlst = st.maxnlst;
   const real buf2 = (st.cutoff + st.buffer) * (st.cutoff + st.buffer);

   const auto* restrict lx = st.x;
   const auto* restrict ly = st.y;
   const auto* restrict lz = st.z;
   auto* restrict xo = st.xold;
   auto* restrict yo = st.yold;
   auto* restrict zo = st.zold;
   auto* restrict nlst = st.nlst;
   auto* restrict lst = st.lst;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(lx,ly,lz,xo,yo,zo,nlst,lst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = lx[i];
      real yi = ly[i];
      real zi = lz[i];
      xo[i] = xi;
      yo[i] = yi;
      zo[i] = zi;

      int ilst = 0;
      #pragma acc loop vector independent
      for (int k = i + 1; k < n; ++k) {
         real xr = xi - lx[k];
         real yr = yi - ly[k];
         real zr = zi - lz[k];
         real r2 = imagen2(xr, yr, zr);
         if (r2 <= buf2) {
            int j;
            #pragma acc atomic capture
            {
               j = ilst;
               ilst += 1;
            }
            lst[i * maxnlst + j] = k;
         }
      }
      nlst[i] = ilst;
   }
}

static void nblistCheck_acc(int n, real lbuf, int* restrict update, const real* restrict x, const real* restrict y,
   const real* restrict z, real* restrict xold, real* restrict yold, real* restrict zold)
{
   const real lbuf2 = (0.5f * lbuf) * (0.5f * lbuf);
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(update,x,y,z,xold,yold,zold)
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real xr = xi - xold[i];
      real yr = yi - yold[i];
      real zr = zi - zold[i];
      real r2 = imagen2(xr, yr, zr);
      update[i] = 0;
      if (r2 >= lbuf2) {
         update[i] = 1;
         xold[i] = xi;
         yold[i] = yi;
         zold[i] = zi;
      }
   }
}

static void nblistUpdate_acc1(NBListUnit nu)
{
   // test sites for displacement exceeding half the buffer

   auto& st = *nu;
   nblistCheck_acc(n, st.buffer, st.update, st.x, st.y, st.z, st.xold, st.yold, st.zold);

   // rebuild the higher numbered neighbors for updated sites

   auto* lst = nu.deviceptr();
   const int maxnlst = st.maxnlst;
   const real buf2 = (st.cutoff + st.buffer) * (st.cutoff + st.buffer);
   const real bufx = (st.cutoff + 2 * st.buffer) * (st.cutoff + 2 * st.buffer);

   #pragma acc kernels async present(lvec1,lvec2,lvec3,recipa,recipb,recipc) deviceptr(lst)
   {
      #pragma acc loop independent
      for (int i = 0; i < n; ++i) {
         if (lst->update[i]) {
            real xi = lst->xold[i];
            real yi = lst->yold[i];
            real zi = lst->zold[i];
            lst->nlst[i] = 0;
            #pragma acc loop seq
            for (int k = i + 1; k < n; ++k) {
               real xr = xi - lst->xold[k];
               real yr = yi - lst->yold[k];
               real zr = zi - lst->zold[k];
               real r2 = imagen2(xr, yr, zr);
               if (r2 <= buf2) {
                  int current = lst->nlst[i];
                  lst->lst[i * maxnlst + current] = k;
                  lst->nlst[i] += 1;
               }
            }
         }
      }

      #pragma acc loop seq
      for (int i = 0; i < n; ++i) {
         if (lst->update[i]) {
            real xi = lst->xold[i];
            real yi = lst->yold[i];
            real zi = lst->zold[i];
            #pragma acc loop seq
            for (int k = 0; k < i - 1; ++k) {
               if (!lst->update[k]) {
                  real xr = xi - lst->xold[k];
                  real yr = yi - lst->yold[k];
                  real zr = zi - lst->zold[k];
                  real r2 = imagen2(xr, yr, zr);
                  if (r2 <= buf2) {
                     for (int j = 0; j < lst->nlst[k]; ++j) {
                        if (lst->lst[k * maxnlst + j] == i)
                           goto label_20;
                     }
                     lst->lst[k * maxnlst + lst->nlst[k]] = i;
                     lst->nlst[k] += 1;
                  label_20:;
                  } else if (r2 <= bufx) {
                     for (int j = 0; j < lst->nlst[k]; ++j) {
                        if (lst->lst[k * maxnlst + j] == i) {
                           lst->nlst[k] -= 1;
                           lst->lst[k * maxnlst + j] = lst->lst[k * maxnlst + lst->nlst[k]];
                           goto label_30;
                        }
                     }
                  label_30:;
                  }
               }
            }
            nblistSort_acc1(&lst->lst[i * maxnlst], lst->nlst[i]);
         }
      }
   }
}
}

namespace tinker {
void nblistBuild_acc(NBListUnit nu)
{
   if (nu->maxnlst == 1) {
      nblistBuildDoubleLoop_acc(nu);
   } else {
      nblistBuild_acc1(nu);
   }
}

void nblistUpdate_acc(NBListUnit nu)
{
   if (nu->maxnlst == 1) {
      // nblistUpdateDoubleLoop_acc();
   } else {
      // nblistBuild_acc1(nu);
      nblistUpdate_acc1(nu);
   }
}

void spatialCheck_acc(int& result, int n, real lbuf, int* restrict update, const real* restrict x,
   const real* restrict y, const real* restrict z, real* restrict xold, real* restrict yold, real* restrict zold)
{
   if (lbuf == 0) {
      result = 1;
      return;
   }

   // 0: do not rebuild; 1: rebuild
   const real lbuf2 = (0.5f * lbuf) * (0.5f * lbuf);
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(update,x,y,z,xold,yold,zold)
   for (int i = 0; i < n; ++i) {
      real xr = x[i] - xold[i];
      real yr = y[i] - yold[i];
      real zr = z[i] - zold[i];
      real r2 = imagen2(xr, yr, zr);
      update[i] = (r2 >= lbuf2 ? 1 : 0);
   }
   #pragma acc parallel loop independent async deviceptr(update)
   for (int i = 0; i < n; ++i) {
      auto rebuild = update[i];
      if (rebuild)
         update[0] = 1;
   }
   int ans;
   darray::copyout(g::q0, 1, &ans, &update[0]);
   waitFor(g::q0);
   result = ans;
}
}
