#include "seq/add.h"
#include "seq/launch.h"

namespace tinker {
__global__
void sumGradient_cu1(int n, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x, const grad_prec* g1y,
   const grad_prec* g1z)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      g0x[i] += g1x[i];
      g0y[i] += g1y[i];
      g0z[i] += g1z[i];
   }
}

void sumGradientV1_cu(grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x, const grad_prec* g1y,
   const grad_prec* g1z)
{
   launch_k1s(g::s0, n, sumGradient_cu1, n, g0x, g0y, g0z, g1x, g1y, g1z);
}

__global__
void sumGradient_cu2(int n, real s, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x,
   const grad_prec* g1y, const grad_prec* g1z)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      auto gxi = toFloatGrad<real>(g1x[i]);
      auto gyi = toFloatGrad<real>(g1y[i]);
      auto gzi = toFloatGrad<real>(g1z[i]);
      g0x[i] += floatTo<grad_prec>(s * gxi);
      g0y[i] += floatTo<grad_prec>(s * gyi);
      g0z[i] += floatTo<grad_prec>(s * gzi);
   }
}

void sumGradientV2_cu(double ss, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x,
   const grad_prec* g1y, const grad_prec* g1z)
{
   real s = ss;
   launch_k1s(g::s0, n, sumGradient_cu2, n, s, g0x, g0y, g0z, g1x, g1y, g1z);
}
}
