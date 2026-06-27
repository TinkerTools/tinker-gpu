#include "md/misc.h"
#include "md/pq.h"
#include "seq/launch.h"
#include "seq/reduce.h"
#include "tool/externfunc.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/units.hh>

namespace tinker {
template <unsigned int B>
__global__
void mdrestSumP_cu(int n, vel_prec* restrict odata, const double* restrict mass, const vel_prec* restrict vx,
   const vel_prec* restrict vy, const vel_prec* restrict vz)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   vel_prec x = 0, y = 0, z = 0;
   for (int i = ithread; i < n; i += stride) {
      auto m = mass[i];
      x += m * vx[i];
      y += m * vy[i];
      z += m * vz[i];
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = x; ty[t] = y; tz[t] = z;                                          __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   if (t == 0) {
      const int b = blockIdx.x;
      odata[3 * b + 0] = tx[t] + tx[t + 1];
      odata[3 * b + 1] = ty[t] + ty[t + 1];
      odata[3 * b + 2] = tz[t] + tz[t + 1];
   }
}

template <int B>
__global__
void mdrestRemoveP_cu(int n, double invtotmass, const vel_prec* restrict idata, vel_prec* restrict vx,
   vel_prec* restrict vy, vel_prec* restrict vz, vel_prec* restrict xout)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   vel_prec x = 0, y = 0, z = 0;
   for (int i = t; i < gridDim.x; i += B) {
      x += idata[3 * i + 0];
      y += idata[3 * i + 1];
      z += idata[3 * i + 2];
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = x; ty[t] = y; tz[t] = z;                                          __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   x = (tx[0] + tx[1]) * invtotmass;
   y = (ty[0] + ty[1]) * invtotmass;
   z = (tz[0] + tz[1]) * invtotmass;
   xout[0] = x;
   xout[1] = y;
   xout[2] = z;
   for (int i = ithread; i < n; i += stride) {
      vx[i] -= x;
      vy[i] -= y;
      vz[i] -= z;
   }
}

void mdrestRemovePbcMomentum_cu(bool copyout, vel_prec& vtot1, vel_prec& vtot2, vel_prec& vtot3)
{
   vel_prec* xout;
   xout = (vel_prec*)dptr_buf;
   auto invtotmass = 1 / molcul::totmass;

   constexpr int HN = 3;
   constexpr int B = 64;
   vel_prec* ptr = &xout[4];
   int grid_siz1 = -4 + gpuGridSize(BLOCK_DIM);
   grid_siz1 /= HN;
   int grid_siz2 = (n + B - 1) / B;
   int ngrid = std::min(grid_siz1, grid_siz2);

   mdrestSumP_cu<B><<<ngrid, B, 0, g::s0>>>(n, ptr, mass, vx, vy, vz);
   mdrestRemoveP_cu<B><<<ngrid, B, 0, g::s0>>>(n, invtotmass, ptr, vx, vy, vz, xout);

   if (copyout) {
      vel_prec v[3];
      darray::copyout(g::q0, 3, v, xout);
      waitFor(g::q0);
      vtot1 = v[0];
      vtot2 = v[1];
      vtot3 = v[2];
   }
}

void mdrest_cu(int istep)
{
   if (not mdstuf::dorest)
      return;
   if ((istep % mdstuf::irest) != 0)
      return;

   // const energy_prec ekcal = units::ekcal;

   // zero out the total mass and overall linear velocity

   auto totmass = molcul::totmass;
   vel_prec vtot1 = 0, vtot2 = 0, vtot3 = 0;

   bool copyout = static_cast<bool>(inform::debug) or not static_cast<bool>(bound::use_bounds);
   mdrestRemovePbcMomentum_cu(copyout, vtot1, vtot2, vtot3);

   // print the translational velocity of the overall system

   mdrestPrintP1(static_cast<bool>(inform::debug), vtot1, vtot2, vtot3, totmass);

   if (not bound::use_bounds) {
      throwExceptionMissingFunction("mdrestRemoveAngularMomentum_cu", __FILE__, __LINE__);
   }
}
}
