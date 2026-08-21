#include "ff/dlmda.h"
#include "seq/add.h"
#include "seq/launch.h"

namespace tinker {
template <class LTYP>
__global__
void adtMixEnergy_cu1(size_t buffer_size, energy_prec weight1, energy_prec dweight1, energy_prec d2weight1,
   const EnergyBufferTraits::type* restrict e0, EnergyBufferTraits::type* restrict e1,
   EnergyBufferTraits::type* restrict dedl, EnergyBufferTraits::type* restrict d2edl2)
{
   for (size_t i = ITHREAD; i < buffer_size; i += STRIDE) {
      auto e0i = toFloatGrad<energy_prec>(e0[i]);
      auto e1i = toFloatGrad<energy_prec>(e1[i]);
      auto diff = e1i - e0i;
      e1[i] = floatTo<EnergyBufferTraits::type>(weight1 * e1i + (1 - weight1) * e0i);
      if CONSTEXPR (eq<LTYP, DLMDA>()) {
         if (dedl)
            dedl[i] += floatTo<EnergyBufferTraits::type>(dweight1 * diff);
         if (d2edl2)
            d2edl2[i] += floatTo<EnergyBufferTraits::type>(d2weight1 * diff);
      }
   }
}

template <class LTYP>
__global__
void adtMixVirial_cu1(size_t size, virial_prec weight1, virial_prec dweight1, const VirialBuffer restrict v0,
   VirialBuffer restrict v1, VirialBuffer restrict dvdl)
{
   for (size_t i = ITHREAD; i < size; i += STRIDE) {
      auto v0i = toFloatGrad<virial_prec>(v0[0][i]);
      auto v1i = toFloatGrad<virial_prec>(v1[0][i]);
      v1[0][i] = floatTo<VirialBufferTraits::type>(weight1 * v1i + (1 - weight1) * v0i);
      if CONSTEXPR (eq<LTYP, DLMDA>())
         if (dvdl)
            dvdl[0][i] += floatTo<VirialBufferTraits::type>(dweight1 * (v1i - v0i));
   }
}

template <class LTYP>
__global__
void adtMixGradient_cu1(int n, real weight1, real dweight1, const grad_prec* restrict gx0,
   const grad_prec* restrict gy0, const grad_prec* restrict gz0, grad_prec* restrict gx1, grad_prec* restrict gy1,
   grad_prec* restrict gz1, grad_prec* restrict dgxdl, grad_prec* restrict dgydl, grad_prec* restrict dgzdl)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      auto x0 = toFloatGrad<real>(gx0[i]);
      auto y0 = toFloatGrad<real>(gy0[i]);
      auto z0 = toFloatGrad<real>(gz0[i]);
      auto x1 = toFloatGrad<real>(gx1[i]);
      auto y1 = toFloatGrad<real>(gy1[i]);
      auto z1 = toFloatGrad<real>(gz1[i]);
      gx1[i] = floatTo<grad_prec>(weight1 * x1 + (1 - weight1) * x0);
      gy1[i] = floatTo<grad_prec>(weight1 * y1 + (1 - weight1) * y0);
      gz1[i] = floatTo<grad_prec>(weight1 * z1 + (1 - weight1) * z0);
      if CONSTEXPR (eq<LTYP, DLMDA>()) {
         if (dgxdl) {
            dgxdl[i] += floatTo<grad_prec>(dweight1 * (x1 - x0));
            dgydl[i] += floatTo<grad_prec>(dweight1 * (y1 - y0));
            dgzdl[i] += floatTo<grad_prec>(dweight1 * (z1 - z0));
         }
      }
   }
}

template <class LTYP>
static void adtMix_cu2(int vers, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl)
{
   if (vers & calc::energy) {
      auto w1 = static_cast<energy_prec>(weight1);
      auto dw1 = static_cast<energy_prec>(dweight1);
      auto d2w1 = static_cast<energy_prec>(d2weight1);
      launch_k1s(g::s0, buffer_size, adtMixEnergy_cu1<LTYP>, buffer_size, w1, dw1, d2w1, e0, e1, dedl, d2edl2);
   }
   if (vers & calc::virial) {
      auto size = buffer_size * VirialBufferTraits::value;
      auto w1 = static_cast<virial_prec>(weight1);
      auto dw1 = static_cast<virial_prec>(dweight1);
      launch_k1s(g::s0, size, adtMixVirial_cu1<LTYP>, size, w1, dw1, v0, v1, dvdl);
   }
   if (vers & calc::grad) {
      auto w1 = static_cast<real>(weight1);
      auto dw1 = static_cast<real>(dweight1);
      launch_k1s(g::s0, n, adtMixGradient_cu1<LTYP>, n, w1, dw1, gx0, gy0, gz0, gx1, gy1, gz1, dgxdl, dgydl,
         dgzdl);
   }
}

void adtMix_cu(int vers, bool do_dlmda, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl)
{
   if (do_dlmda)
      adtMix_cu2<DLMDA>(vers, n, buffer_size, weight1, dweight1, d2weight1, e0, e1, dedl, d2edl2, v0, v1, dvdl,
         gx0, gy0, gz0, gx1, gy1, gz1, dgxdl, dgydl, dgzdl);
   else
      adtMix_cu2<NON_DLMDA>(vers, n, buffer_size, weight1, dweight1, d2weight1, e0, e1, dedl, d2edl2, v0, v1,
         dvdl, gx0, gy0, gz0, gx1, gy1, gz1, dgxdl, dgydl, dgzdl);
}
}
