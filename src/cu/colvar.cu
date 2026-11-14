#include "ff/atom.h"
#include "ff/box.h"
#include "ff/colvar.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/molecule.h"
#include "seq/colvar.h"
#include "seq/launch.h"
#include "tool/darray.h"

namespace tinker {
// One thread per entry of satom adds the entry's share to the fixed-point
// center sum of its set. A warp whose lanes all belong to one set adds its sum
// with a single set of atomics, so a large set does not pile every thread onto
// the same three addresses.
__global__
static void colvarSum_cu1(int nsat, grad_prec (*restrict ssum)[3], const int* restrict sown,
   const int (*restrict srange)[2], const int* restrict satom, const real* restrict sminv, const real* restrict x,
   const real* restrict y, const real* restrict z, const double* restrict mass, const int* restrict molec,
   TINKER_IMAGE_PARAMS)
{
   // every lane of a warp takes the same trips through the loop
   const int npad = (nsat + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);
   for (int j = ITHREAD; j < npad; j += STRIDE) {
      int s = -1;
      real dx = 0, dy = 0, dz = 0;
      if (j < nsat)
         colvarSetTerm(s, dx, dy, dz, j, sown, srange, satom, sminv, x, y, z, mass, molec, TINKER_IMAGE_ARGS);
      const int s0 = __shfl_sync(ALL_LANES, s, 0);
      if (__all_sync(ALL_LANES, s == s0)) {
         #pragma unroll
         for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            dx += __shfl_down_sync(ALL_LANES, dx, off);
            dy += __shfl_down_sync(ALL_LANES, dy, off);
            dz += __shfl_down_sync(ALL_LANES, dz, off);
         }
         if (ilane != 0)
            s = -1;
      }
      if (s >= 0) {
         atomic_add(dx, ssum[s], 0);
         atomic_add(dy, ssum[s], 1);
         atomic_add(dz, ssum[s], 2);
      }
   }
}

// One thread per gradient slot finishes the CV of the slot and writes the
// slot, adding the LAMBDA-CV restraint of a coupled CV; launched with no more
// threads than the energy buffer has entries, so each adds into its own.
template <class Ver>
__global__
static void colvarSlot_cu1(int nslot, const int* restrict gcv, const int* restrict type, real* restrict val,
   real (*restrict w)[9], real (*restrict grad)[3], const grad_prec (*restrict ssum)[3], const int (*restrict iarg)[4],
   const int (*restrict slot)[2], const int (*restrict srange)[2], const int* restrict satom,
   const real* restrict sminv, const real* restrict x, const real* restrict y, const real* restrict z,
   const double* restrict mass, const int* restrict molec, TINKER_IMAGE_PARAMS, //
   const int* restrict gatom, const real (*restrict couple)[3], real lg, real lg1, real lg2, EnergyBuffer restrict ecv,
   VirialBuffer restrict vir_ecv, grad_prec* restrict decvx, grad_prec* restrict decvy, grad_prec* restrict decvz,
   EnergyBuffer restrict dl1, EnergyBuffer restrict dl2, VirialBuffer restrict dvirdl, grad_prec* restrict dfdlx,
   grad_prec* restrict dfdly, grad_prec* restrict dfdlz)
{
   const int ithread = ITHREAD;
   for (int g = ithread; g < nslot; g += STRIDE) {
      real q, gsx, gsy, gsz;
      colvarSlot1(q, gsx, gsy, gsz, g, gcv, type, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass,
         molec, TINKER_IMAGE_ARGS);
      colvarCouple<Ver>(g, ithread, q, gsx, gsy, gsz, gcv, slot, gatom, couple, w, lg, lg1, lg2, ecv, vir_ecv, decvx,
         decvy, decvz, dl1, dl2, dvirdl, dfdlx, dfdly, dfdlz);
   }
}

template <class Ver>
static void ecolvar_cu2()
{
   const auto& c = colvar;
   darray::zero(g::q0, c.nset, c.ssum);
   launch_k1s(g::s0, c.nsat, colvarSum_cu1, c.nsat, c.ssum, c.sown, c.srange, c.satom, c.sminv, x, y, z, mass,
      molecule.molecule, TINKER_IMAGE_ARGS);
   // the collective variable sub-lambda and its first two derivatives
   const real lg = cvlam, lg1 = dcvldlmda, lg2 = d2cvldlmda2;
   launch_k1b(g::s0, c.nslot, colvarSlot_cu1<Ver>, c.nslot, c.gcv, c.type, c.val, c.w, c.grad, c.ssum, c.iarg, c.slot,
      c.srange, c.satom, c.sminv, x, y, z, mass, molecule.molecule, TINKER_IMAGE_ARGS, //
      c.gatom, c.couple, lg, lg1, lg2, ecv, vir_ecv, decvx, decvy, decvz, decvdl_buf, d2ecvdl2_buf, dvirdl_buf, dfdlx,
      dfdly, dfdlz);
}

void ecolvar_cu(int vers)
{
   if (vers == calc::v0)
      ecolvar_cu2<calc::V0>();
   else if (vers == calc::v1)
      ecolvar_cu2<calc::V1>();
   else if (vers == calc::v3)
      ecolvar_cu2<calc::V3>();
   else if (vers == calc::v4)
      ecolvar_cu2<calc::V4>();
   else if (vers == calc::v5)
      ecolvar_cu2<calc::V5>();
   else if (vers == calc::v6)
      ecolvar_cu2<calc::V6>();
   else if (vers == calc::v7)
      ecolvar_cu2<calc::V7>();
   else if (vers == calc::v8)
      ecolvar_cu2<calc::V8>();
   else if (vers == calc::v9)
      ecolvar_cu2<calc::V9>();
   else if (vers == calc::v10)
      ecolvar_cu2<calc::V10>();
}
}
