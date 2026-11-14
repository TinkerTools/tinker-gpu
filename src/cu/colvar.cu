#include "ff/atom.h"
#include "ff/box.h"
#include "ff/colvar.h"
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

__global__
static void colvarSlot_cu1(int nslot, const int* restrict gcv, const int* restrict type, real* restrict val,
   real (*restrict w)[9], real (*restrict grad)[3], const grad_prec (*restrict ssum)[3], const int (*restrict iarg)[4],
   const int (*restrict slot)[2], const int (*restrict srange)[2], const int* restrict satom,
   const real* restrict sminv, const real* restrict x, const real* restrict y, const real* restrict z,
   const double* restrict mass, const int* restrict molec, TINKER_IMAGE_PARAMS)
{
   for (int g = ITHREAD; g < nslot; g += STRIDE)
      colvarSlot1(g, gcv, type, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass, molec,
         TINKER_IMAGE_ARGS);
}

void colvarEval_cu()
{
   const auto& c = colvar;
   darray::zero(g::q0, c.nset, c.ssum);
   launch_k1s(g::s0, c.nsat, colvarSum_cu1, c.nsat, c.ssum, c.sown, c.srange, c.satom, c.sminv, x, y, z, mass,
      molecule.molecule, TINKER_IMAGE_ARGS);
   launch_k1s(g::s0, c.nslot, colvarSlot_cu1, c.nslot, c.gcv, c.type, c.val, c.w, c.grad, c.ssum, c.iarg, c.slot,
      c.srange, c.satom, c.sminv, x, y, z, mass, molecule.molecule, TINKER_IMAGE_ARGS);
}
}
