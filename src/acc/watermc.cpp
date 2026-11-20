#include "ff/watermccheap.h"
#include "seq/watermccheap.h"

namespace tinker {
void waterMcCheap_acc(const WaterMcCheapTerms& t0, const WaterMcCheapDev& p0, const WaterMcCheapBatch& b)
{
   // copies by value, which firstprivate takes to the device; of a reference
   // it would take the host address
   const WaterMcCheapTerms t = t0;
   const WaterMcCheapDev p = p0;
   const int nori = b.nori;
   const int* beg = b.beg;
   const int* end = b.end;
   const real* site = b.site;
   const int* nbj = b.nbj;
   const int* nbmc = b.nbmc;
   const real* nbr = b.nbr;
   double* u = b.u;
   int* hit = b.hit;
   #pragma acc parallel loop gang independent async\
               deviceptr(beg,end,site,nbj,nbmc,nbr,u,hit) firstprivate(t,p)
   for (int o = 0; o < nori; ++o) {
      const real* ws = site + (size_t)o * 3 * WMC_NSITE;
      double acc = 0;
      int h = 0;
      #pragma acc loop vector reduction(+:acc) reduction(max:h)
      for (int k = beg[o]; k < end[o]; ++k) {
         const real* nb = nbr + (size_t)k * WMC_NNB;
         const int j = nbj[k], mc = nbmc[k];
         int hk = 0;
         double e = 0;
         #pragma acc loop seq
         for (int w = 0; w < 3; ++w)
            e += waterMcCheapPair(t, p, ws + w * WMC_NSITE, t.a[w], j, mc, nb, hk);
         acc += e;
         h = h > hk ? h : hk;
      }
      u[o] = acc;
      hit[o] = h;
   }
}
}
