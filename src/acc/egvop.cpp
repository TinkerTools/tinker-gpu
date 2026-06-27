#include "ff/atom.h"
#include "seq/add.h"

namespace tinker {
void scaleGradient_acc(double scale, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z)
{
   real s = scale;
#if TINKER_DETERMINISTIC_FORCE
   #pragma acc parallel loop independent async\
           deviceptr(g0x,g0y,g0z)
   for (int i = 0; i < n; ++i) {
      real dx = s * fixedTo<real>(g0x[i]);
      real dy = s * fixedTo<real>(g0y[i]);
      real dz = s * fixedTo<real>(g0z[i]);
      g0x[i] = floatTo<grad_prec>(dx);
      g0y[i] = floatTo<grad_prec>(dy);
      g0z[i] = floatTo<grad_prec>(dz);
   }
#else
   #pragma acc parallel loop independent async\
           deviceptr(g0x,g0y,g0z)
   for (int i = 0; i < n; ++i) {
      g0x[i] *= s;
      g0y[i] *= s;
      g0z[i] *= s;
   }
#endif
}

void sumGradientV1_acc(grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x, const grad_prec* g1y,
   const grad_prec* g1z)
{
   #pragma acc parallel loop independent async\
           deviceptr(g0x,g0y,g0z,g1x,g1y,g1z)
   for (int i = 0; i < n; ++i) {
      g0x[i] += g1x[i];
      g0y[i] += g1y[i];
      g0z[i] += g1z[i];
   }
}

void sumGradientV2_acc(double ss, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x,
   const grad_prec* g1y, const grad_prec* g1z)
{
   real s = ss;
#if TINKER_DETERMINISTIC_FORCE
   #pragma acc parallel loop independent async\
           deviceptr(g0x,g0y,g0z,g1x,g1y,g1z)
   for (int i = 0; i < n; ++i) {
      real dx = s * fixedTo<real>(g1x[i]);
      real dy = s * fixedTo<real>(g1y[i]);
      real dz = s * fixedTo<real>(g1z[i]);
      g0x[i] += floatTo<grad_prec>(dx);
      g0y[i] += floatTo<grad_prec>(dy);
      g0z[i] += floatTo<grad_prec>(dz);
   }
#else
   #pragma acc parallel loop independent async\
           deviceptr(g0x,g0y,g0z,g1x,g1y,g1z)
   for (int i = 0; i < n; ++i) {
      g0x[i] += s * g1x[i];
      g0y[i] += s * g1y[i];
      g0z[i] += s * g1z[i];
   }
#endif
}
}
