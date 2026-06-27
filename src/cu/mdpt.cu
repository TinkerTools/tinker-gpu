#include "ff/molecule.h"
#include "md/pq.h"
#include "seq/launch.h"
#include "seq/reduce.h"
#include "tool/error.h"
#include <tinker/detail/units.hh>

namespace tinker {
// velocity to ekin[3][3] (actually ekin[6])
template <unsigned int B>
__global__
void velocityToEkin_cu(energy_prec* out, const vel_prec* restrict vx, const vel_prec* restrict vy,
   const vel_prec* restrict vz, const double* restrict mass, int n, energy_prec ekcal_inv)
{
   constexpr int HN = 6;
   __shared__ energy_prec sd[HN][B];
   unsigned int t = threadIdx.x;
   #pragma unroll
   for (int j = 0; j < HN; ++j)
      sd[j][t] = 0;
   for (int i = t + blockIdx.x * B; i < n; i += B * gridDim.x) {
      energy_prec term = 0.5f * mass[i] * ekcal_inv;
      sd[0][t] += term * vx[i] * vx[i]; // exx
      sd[1][t] += term * vy[i] * vy[i]; // eyy
      sd[2][t] += term * vz[i] * vz[i]; // ezz
      sd[3][t] += term * vx[i] * vy[i]; // exy
      sd[4][t] += term * vy[i] * vz[i]; // eyz
      sd[5][t] += term * vz[i] * vx[i]; // ezx
   }
   __syncthreads();

   using Op = OpPlus<energy_prec>;
   Op op;
   static_assert(B <= 512, "");
   // clang-format off
   if (B >= 512) { if (t < 256) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 256]); } __syncthreads(); }
   if (B >= 256) { if (t < 128) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 128]); } __syncthreads(); }
   if (B >= 128) { if (t < 64 ) { _Pragma("unroll") for (int j = 0; j < HN; ++j) sd[j][t] = op(sd[j][t], sd[j][t + 64 ]); } __syncthreads(); }
   if (t < 32  ) warp_reduce2<energy_prec, HN, B, Op>(sd, t, op);
   // clang-format on
   if (t == 0)
      #pragma unroll
      for (int j = 0; j < HN; ++j)
         out[blockIdx.x * HN + j] = sd[j][0];
}

void kineticEnergy_cu(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n, const double* mass,
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   cudaStream_t st = g::s0;
   constexpr int HN = 6;
   energy_prec* dptr = reinterpret_cast<energy_prec*>(dptr_buf);
   energy_prec(*dptr6)[HN] = reinterpret_cast<energy_prec(*)[HN]>(dptr_buf);
   energy_prec* hptr = reinterpret_cast<energy_prec*>(pinned_buf);
   int grid_siz1 = gpuGridSize(BLOCK_DIM);
   grid_siz1 = grid_siz1 / HN;
   int grid_siz2 = (n + BLOCK_DIM - 1) / BLOCK_DIM;
   int grid_size = std::min(grid_siz1, grid_siz2);
   const energy_prec ekcal_inv = 1.0 / units::ekcal;
   velocityToEkin_cu<BLOCK_DIM><<<grid_size, BLOCK_DIM, 0, st>>>(dptr, vx, vy, vz, mass, n, ekcal_inv);
   reduce2<energy_prec, BLOCK_DIM, HN, HN, OpPlus<energy_prec>><<<1, BLOCK_DIM, 0, st>>>(dptr6, dptr6, grid_size);
   check_rt(cudaMemcpyAsync(hptr, dptr, HN * sizeof(energy_prec), cudaMemcpyDeviceToHost, st));
   check_rt(cudaStreamSynchronize(st));
   energy_prec exx = hptr[0];
   energy_prec eyy = hptr[1];
   energy_prec ezz = hptr[2];
   energy_prec exy = hptr[3];
   energy_prec eyz = hptr[4];
   energy_prec ezx = hptr[5];

   ekin_out[0][0] = exx;
   ekin_out[0][1] = exy;
   ekin_out[0][2] = ezx;
   ekin_out[1][0] = exy;
   ekin_out[1][1] = eyy;
   ekin_out[1][2] = eyz;
   ekin_out[2][0] = ezx;
   ekin_out[2][1] = eyz;
   ekin_out[2][2] = ezz;
   eksum_out = exx + eyy + ezz;
}
}

namespace tinker {
__global__
void scaleBarostatAtomMove_cu1(int n, pos_prec scalex, pos_prec scaley, pos_prec scalez, //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      xpos[i] *= scalex;
      ypos[i] *= scaley;
      zpos[i] *= scalez;
   }
}

__global__
void scaleBarostatAtomMoveAniso_cu1(pos_prec a00, pos_prec a01, pos_prec a02, //
   pos_prec a10, pos_prec a11, pos_prec a12,                                  //
   pos_prec a20, pos_prec a21, pos_prec a22, int n,                           //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      pos_prec xk = xpos[i];
      pos_prec yk = ypos[i];
      pos_prec zk = zpos[i];
      xpos[i] = xk * a00 + yk * a01 + zk * a02;
      ypos[i] = xk * a10 + yk * a11 + zk * a12;
      zpos[i] = xk * a20 + yk * a21 + zk * a22;
   }
}

void scaleBarostatAtomMove_cu(double scalex, double scaley, double scalez)
{
   launch_k1s(g::s0, n, scaleBarostatAtomMove_cu1, //
      n, scalex, scaley, scalez, xpos, ypos, zpos);
}

void scaleBarostatAtomMoveAniso_cu(const double ascale[3][3])
{
   launch_k1s(g::s0, n, scaleBarostatAtomMoveAniso_cu1, //
      ascale[0][0], ascale[0][1], ascale[0][2],         //
      ascale[1][0], ascale[1][1], ascale[1][2],         //
      ascale[2][0], ascale[2][1], ascale[2][2], n,      //
      xpos, ypos, zpos);
}

__global__
void scaleVelocity_cu1(int n, vel_prec scalex, vel_prec scaley, vel_prec scalez, //
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      vx[i] *= scalex;
      vy[i] *= scaley;
      vz[i] *= scalez;
   }
}

__global__
void scaleVelocityAniso_cu1(vel_prec a00, vel_prec a01, vel_prec a02, //
   vel_prec a10, vel_prec a11, vel_prec a12,                          //
   vel_prec a20, vel_prec a21, vel_prec a22, int n,                   //
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      vel_prec xk = vx[i];
      vel_prec yk = vy[i];
      vel_prec zk = vz[i];
      vx[i] = xk * a00 + yk * a01 + zk * a02;
      vy[i] = xk * a10 + yk * a11 + zk * a12;
      vz[i] = xk * a20 + yk * a21 + zk * a22;
   }
}

void scaleVelocity_cu(double scalex, double scaley, double scalez)
{
   launch_k1s(g::s0, n, scaleVelocity_cu1, //
      n, scalex, scaley, scalez, vx, vy, vz);
}

void scaleVelocityAniso_cu(const double ascale[3][3])
{
   launch_k1s(g::s0, n, scaleVelocityAniso_cu1, //
      ascale[0][0], ascale[0][1], ascale[0][2], //
      ascale[1][0], ascale[1][1], ascale[1][2], //
      ascale[2][0], ascale[2][1], ascale[2][2], n, vx, vy, vz);
}
}

namespace tinker {
__global__
void monteCarloMolMove_cu1(pos_prec pos_scale, int nmol,                      //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos, //
   const int (*restrict imol)[2], const int* restrict kmol, const double* restrict mass, const double* restrict molmass)
{
   for (int i = ITHREAD; i < nmol; i += STRIDE) {
      pos_prec xcm = 0, ycm = 0, zcm = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         auto weigh = mass[k];
         xcm += xpos[k] * weigh;
         ycm += ypos[k] * weigh;
         zcm += zpos[k] * weigh;
      }
      pos_prec term = pos_scale / molmass[i];
      pos_prec xmove, ymove, zmove;
      xmove = term * xcm;
      ymove = term * ycm;
      zmove = term * zcm;
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xmove;
         ypos[k] += ymove;
         zpos[k] += zmove;
      }
   }
}

void monteCarloMolMove_cu(double scale)
{
   auto nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;
   pos_prec pos_scale = scale - 1;

   launch_k1s(g::s0, nmol, monteCarloMolMove_cu1, //
      pos_scale, nmol,                            //
      xpos, ypos, zpos,                           //
      imol, kmol, mass, molmass);
}

__global__
void monteCarloMolMoveAniso_cu1(pos_prec a00, pos_prec a01, pos_prec a02, //
   pos_prec a10, pos_prec a11, pos_prec a12,                              //
   pos_prec a20, pos_prec a21, pos_prec a22, int nmol,                    //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos, const int (*restrict imol)[2],
   const int* restrict kmol, //
   const double* restrict mass, const double* restrict molmass)
{
   for (int i = ITHREAD; i < nmol; i += STRIDE) {
      pos_prec xcm = 0, ycm = 0, zcm = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         auto weigh = mass[k];
         xcm += xpos[k] * weigh;
         ycm += ypos[k] * weigh;
         zcm += zpos[k] * weigh;
      }
      pos_prec inv_mass = 1 / molmass[i];
      xcm *= inv_mass;
      ycm *= inv_mass;
      zcm *= inv_mass;
      pos_prec xmove = xcm * a00 + ycm * a01 + zcm * a02;
      pos_prec ymove = xcm * a10 + ycm * a11 + zcm * a12;
      pos_prec zmove = xcm * a20 + ycm * a21 + zcm * a22;
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xmove;
         ypos[k] += ymove;
         zpos[k] += zmove;
      }
   }
}

void monteCarloMolMoveAniso_cu(const double ascale[3][3])
{
   auto nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;
   pos_prec a00 = ascale[0][0] - 1;
   pos_prec a01 = ascale[0][1];
   pos_prec a02 = ascale[0][2];
   pos_prec a10 = ascale[1][0];
   pos_prec a11 = ascale[1][1] - 1;
   pos_prec a12 = ascale[1][2];
   pos_prec a20 = ascale[2][0];
   pos_prec a21 = ascale[2][1];
   pos_prec a22 = ascale[2][2] - 1;

   launch_k1s(g::s0, nmol, monteCarloMolMoveAniso_cu1,   //
      a00, a01, a02, a10, a11, a12, a20, a21, a22, nmol, //
      xpos, ypos, zpos, imol, kmol, mass, molmass);
}
}
