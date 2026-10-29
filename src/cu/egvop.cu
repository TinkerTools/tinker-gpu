#include "seq/add.h"
#include "seq/launch.h"

namespace tinker {
__global__
static void sumEnergyBuffer_cu1(size_t size, EnergyBuffer restrict dst,
   const EnergyBufferTraits::type* restrict src)
{
   for (size_t i = ITHREAD; i < size; i += STRIDE)
      dst[i] += src[i];
}

void sumEnergyBuffer_cu(size_t size, EnergyBuffer dst, const EnergyBufferTraits::type* src)
{
   launch_k1s(g::s0, size, sumEnergyBuffer_cu1, size, dst, src);
}

__global__
static void sumVirialBuffer_cu1(size_t size, VirialBuffer restrict dst, const VirialBuffer restrict src)
{
   for (size_t i = ITHREAD; i < size; i += STRIDE)
      dst[0][i] += src[0][i];
}

void sumVirialBuffer_cu(size_t size, VirialBuffer dst, const VirialBuffer src)
{
   launch_k1s(g::s0, size, sumVirialBuffer_cu1, size, dst, src);
}

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
