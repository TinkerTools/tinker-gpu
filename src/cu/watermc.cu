#include "ff/watermccheap.h"
#include "seq/launch.h"
#include "seq/watermccheap.h"
#include "tool/error.h"
#include <algorithm>

namespace tinker {
// One block sums one orientation, in a fixed order, so that a hop comes out
// the same from run to run. (The hop back finds the same structure from
// multipoles rotated anew, so the two agree to round-off, not to the bit.)
template <int BD>
__global__
void waterMcCheap_cu1(WaterMcCheapTerms t, WaterMcCheapDev p, int nori, const int* restrict beg,
   const int* restrict end, const real* restrict site, const int* restrict nbj, const int* restrict nbmc,
   const real* restrict nbr, double* restrict u, int* restrict hit)
{
   __shared__ double su[BD];
   for (int o = blockIdx.x; o < nori; o += gridDim.x) {
      const real* ws = site + (size_t)o * 3 * WMC_NSITE;
      double acc = 0;
      int h = 0;
      for (int k = beg[o] + threadIdx.x; k < end[o]; k += BD) {
         const real* nb = nbr + (size_t)k * WMC_NNB;
         const int j = nbj[k], mc = nbmc[k];
         for (int w = 0; w < 3; ++w)
            acc += waterMcCheapPair(t, p, ws + w * WMC_NSITE, t.a[w], j, mc, nb, h);
      }
      su[threadIdx.x] = acc;
      const int hhit = __syncthreads_or(h == WMC_HIT);
      const int hany = __syncthreads_or(h != WMC_CLEAR);
      h = hhit ? WMC_HIT : (hany ? WMC_EDGE : WMC_CLEAR);
      for (int s = BD / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s)
            su[threadIdx.x] += su[threadIdx.x + s];
         __syncthreads();
      }
      if (threadIdx.x == 0) {
         u[o] = su[0];
         hit[o] = h;
      }
      __syncthreads();
   }
}

void waterMcCheap_cu(const WaterMcCheapTerms& t, const WaterMcCheapDev& p, const WaterMcCheapBatch& b)
{
   if (b.nori == 0)
      return;
   constexpr int BD = 64;
   const int ngrid = std::min(b.nori, 65535);
   waterMcCheap_cu1<BD><<<ngrid, BD, 0, g::s0>>>(t, p, b.nori, b.beg, b.end, b.site, b.nbj, b.nbmc, b.nbr, b.u, b.hit);
}
}
