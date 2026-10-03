#include "md/misc.h"
#include "md/pq.h"
#include "seq/launch.h"
#include "seq/reduce.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/ioprint.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>

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

// center of mass coordinates of the overall system, times the total mass
template <unsigned int B>
__global__
void mdrestSumCom_cu(pos_prec* out, int n, const double* restrict mass, const pos_prec* restrict xpos,
   const pos_prec* restrict ypos, const pos_prec* restrict zpos)
{
   constexpr int HN = 3;
   __shared__ pos_prec sd[HN][B];
   unsigned int t = threadIdx.x;
   #pragma unroll
   for (int j = 0; j < HN; ++j)
      sd[j][t] = 0;
   for (int i = t + blockIdx.x * B; i < n; i += B * gridDim.x) {
      auto weigh = mass[i];
      sd[0][t] += xpos[i] * weigh;
      sd[1][t] += ypos[i] * weigh;
      sd[2][t] += zpos[i] * weigh;
   }
   __syncthreads();

   using Op = OpPlus<pos_prec>;
   Op op;
   static_assert(B <= 512, "");
   // clang-format off
   if (B >= 512) { if (t < 256) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 256]); } __syncthreads(); }
   if (B >= 256) { if (t < 128) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 128]); } __syncthreads(); }
   if (B >= 128) { if (t < 64 ) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 64 ]); } __syncthreads(); }
   if (t < 32  ) warp_reduce2<pos_prec, HN, B, Op>(sd, t, op);
   // clang-format on
   if (t == 0)
      #pragma unroll
      for (int j = 0; j < HN; ++j)
         out[blockIdx.x * HN + j] = sd[j][0];
}

// angular momentum (mang[3]) and moment of inertia (xx, xy, xz, yy, yz, zz)
// about the center of mass
template <unsigned int B>
__global__
void mdrestSumAngInertia_cu(vel_prec* out, int n, pos_prec xtot, pos_prec ytot, pos_prec ztot,
   const double* restrict mass, const pos_prec* restrict xpos, const pos_prec* restrict ypos,
   const pos_prec* restrict zpos, const vel_prec* restrict vx, const vel_prec* restrict vy,
   const vel_prec* restrict vz)
{
   constexpr int HN = 9;
   __shared__ vel_prec sd[HN][B];
   unsigned int t = threadIdx.x;
   #pragma unroll
   for (int j = 0; j < HN; ++j)
      sd[j][t] = 0;
   for (int i = t + blockIdx.x * B; i < n; i += B * gridDim.x) {
      auto weigh = mass[i];
      pos_prec xdel = xpos[i] - xtot;
      pos_prec ydel = ypos[i] - ytot;
      pos_prec zdel = zpos[i] - ztot;
      sd[0][t] += (ydel * vz[i] - zdel * vy[i]) * weigh;
      sd[1][t] += (zdel * vx[i] - xdel * vz[i]) * weigh;
      sd[2][t] += (xdel * vy[i] - ydel * vx[i]) * weigh;
      sd[3][t] += xdel * xdel * weigh;
      sd[4][t] += xdel * ydel * weigh;
      sd[5][t] += xdel * zdel * weigh;
      sd[6][t] += ydel * ydel * weigh;
      sd[7][t] += ydel * zdel * weigh;
      sd[8][t] += zdel * zdel * weigh;
   }
   __syncthreads();

   using Op = OpPlus<vel_prec>;
   Op op;
   static_assert(B <= 512, "");
   // clang-format off
   if (B >= 512) { if (t < 256) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 256]); } __syncthreads(); }
   if (B >= 256) { if (t < 128) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 128]); } __syncthreads(); }
   if (B >= 128) { if (t < 64 ) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 64 ]); } __syncthreads(); }
   if (t < 32  ) warp_reduce2<vel_prec, HN, B, Op>(sd, t, op);
   // clang-format on
   if (t == 0)
      #pragma unroll
      for (int j = 0; j < HN; ++j)
         out[blockIdx.x * HN + j] = sd[j][0];
}

__global__
void mdrestRemoveAngularMomentum_cu1(int n, pos_prec xtot, pos_prec ytot, pos_prec ztot, vel_prec vang0,
   vel_prec vang1, vel_prec vang2, const pos_prec* restrict xpos, const pos_prec* restrict ypos,
   const pos_prec* restrict zpos, vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      pos_prec xdel = xpos[i] - xtot;
      pos_prec ydel = ypos[i] - ytot;
      pos_prec zdel = zpos[i] - ztot;
      vx[i] = vx[i] - vang1 * zdel + vang2 * ydel;
      vy[i] = vy[i] - vang2 * xdel + vang0 * zdel;
      vz[i] = vz[i] - vang0 * ydel + vang1 * xdel;
   }
}

// reduces the per-block partials in dptr_buf and copies the HN sums out
template <class T, int HN>
static void mdrestReducePartials_cu(T (&ans)[HN], int grid_size)
{
   cudaStream_t st = g::s0;
   T(*dptrh)[HN] = reinterpret_cast<T(*)[HN]>(dptr_buf);
   T* hptr = reinterpret_cast<T*>(pinned_buf);
   reduce2<T, BLOCK_DIM, HN, HN, OpPlus<T>><<<1, BLOCK_DIM, 0, st>>>(dptrh, dptrh, grid_size);
   check_rt(cudaMemcpyAsync(hptr, dptr_buf, HN * sizeof(T), cudaMemcpyDeviceToHost, st));
   check_rt(cudaStreamSynchronize(st));
   for (int j = 0; j < HN; ++j)
      ans[j] = hptr[j];
}

// Must run after the translation of the overall system has been removed, so
// the angular momentum about the center of mass needs no correction for it.
static void mdrestRemoveAngularMomentum_cu()
{
   cudaStream_t st = g::s0;
   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   grid_siz1 = grid_siz1 / 9; // limited by the output buffer
   int grid_siz2 = (n + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   const energy_prec ekcal = units::ekcal;
   auto totmass = molcul::totmass;

   // find the center of mass coordinates of the overall system

   pos_prec rtot[3];
   mdrestSumCom_cu<BLOCK_DIM><<<grid_size, BLOCK_DIM, 0, st>>>((pos_prec*)dptr_buf, n, mass, xpos, ypos, zpos);
   mdrestReducePartials_cu(rtot, grid_size);
   pos_prec xtot = rtot[0] / totmass;
   pos_prec ytot = rtot[1] / totmass;
   pos_prec ztot = rtot[2] / totmass;

   // compute the angular momentum and the moment of inertia tensor

   vel_prec s[9];
   mdrestSumAngInertia_cu<BLOCK_DIM><<<grid_size, BLOCK_DIM, 0, st>>>(
      (vel_prec*)dptr_buf, n, xtot, ytot, ztot, mass, xpos, ypos, zpos, vx, vy, vz);
   mdrestReducePartials_cu(s, grid_size);
   vel_prec mang1 = s[0], mang2 = s[1], mang3 = s[2];
   pos_prec xx = s[3], xy = s[4], xz = s[5], yy = s[6], yz = s[7], zz = s[8];

   double tensor[3][3];
   double eps = (n <= 2 ? 0.000001 : 0);
   tensor[0][0] = yy + zz + eps;
   tensor[0][1] = -xy;
   tensor[0][2] = -xz;
   tensor[1][0] = -xy;
   tensor[1][1] = xx + zz + eps;
   tensor[1][2] = -yz;
   tensor[2][0] = -xz;
   tensor[2][1] = -yz;
   tensor[2][2] = xx + yy + eps;

   int ndim = 3;
   tinker_f_invert(&ndim, &tensor[0][0]);

   // compute angular velocity and rotational kinetic energy

   vel_prec vang[3];
   for (int i = 0; i < 3; ++i) {
      vang[i] = tensor[0][i] * mang1 + tensor[1][i] * mang2 + tensor[2][i] * mang3;
   }
   energy_prec erot = vang[0] * mang1 + vang[1] * mang2 + vang[2] * mang3;
   erot *= (0.5f / ekcal);

   // eliminate any rotation about the system center of mass

   launch_k1s(st, n, mdrestRemoveAngularMomentum_cu1, //
      n, xtot, ytot, ztot, vang[0], vang[1], vang[2], xpos, ypos, zpos, vx, vy, vz);

   // print the angular velocity of the overall system

   if (inform::debug) {
      print(stdout,
         " System Angular Velocity : %12.2e%12.2e%12.2e\n Rotational "
         "Kinetic Energy :%13s%12.4f Kcal/mole\n",
         vang[0], vang[1], vang[2], "", erot);
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
      mdrestRemoveAngularMomentum_cu();
   }
}
}
