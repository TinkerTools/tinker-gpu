#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "seq/add.h"
#include "seq/emrecip.h"
#include "seq/launch.h"

namespace tinker {
template <bool do_e, bool do_g, bool do_v>
__global__
void empoleEwaldRecipDlmdaGeneric_cu1(int n, real f,                                          //
   EnergyBuffer restrict em, EnergyBuffer restrict demdl, EnergyBuffer restrict d2emdl2, //
   VirialBuffer restrict vir_em, VirialBuffer restrict demvirdl,                       //
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz,       //
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz, //
   real* restrict trqx, real* restrict trqy, real* restrict trqz,                      //
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz,                //
   const real (*restrict cmp)[10], const real (*restrict fmp)[10],                     //
   const real (*restrict cphi)[10], const real (*restrict fphi)[20],                   //
   const real (*restrict dlcmp)[10], const real (*restrict dlfmp)[10],                 //
   const real (*restrict dlcphi)[10], const real (*restrict dlfphi)[20],               //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e, dle, d2le;
      real f1, f2, f3;
      real a1, a2, a3;
      real b1, b2, b3;
      real unused;

      emrecipEnergyForceAtomI<do_e, do_g>(i, fmp, fphi, e, f1, f2, f3);
      emrecipEnergyForceAtomI<do_e, do_g>(i, dlfmp, fphi, dle, a1, a2, a3);
      emrecipEnergyForceAtomI<false, do_g>(i, fmp, dlfphi, unused, b1, b2, b3);
      if CONSTEXPR (do_e) {
         emrecipEnergyForceAtomI<true, false>(i, dlfmp, dlfphi, d2le, unused, unused, unused);
         atomic_add(0.5f * e * f, em, ithread);
         atomic_add(dle * f, demdl, ithread);
         atomic_add(d2le * f, d2emdl2, ithread);
      }

      if CONSTEXPR (do_g) {
         real dlf1 = a1 + b1, dlf2 = a2 + b2, dlf3 = a3 + b3;
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;
         dlf1 *= nfft1;
         dlf2 *= nfft2;
         dlf3 *= nfft3;

         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;
         atomic_add(h1 * f, demx, i);
         atomic_add(h2 * f, demy, i);
         atomic_add(h3 * f, demz, i);

         real dlh1 = recipa.x * dlf1 + recipb.x * dlf2 + recipc.x * dlf3;
         real dlh2 = recipa.y * dlf1 + recipb.y * dlf2 + recipc.y * dlf3;
         real dlh3 = recipa.z * dlf1 + recipb.z * dlf2 + recipc.z * dlf3;
         atomic_add(dlh1 * f, dfmdlx, i);
         atomic_add(dlh2 * f, dfmdly, i);
         atomic_add(dlh3 * f, dfmdlz, i);

         // resolve site torques then increment forces and virial
         real tem[3], t1[3], t2[3];
         emrecipTorqueAtomI(i, cmp, cphi, tem);
         atomic_add(tem[0] * f, trqx, i);
         atomic_add(tem[1] * f, trqy, i);
         atomic_add(tem[2] * f, trqz, i);

         emrecipTorqueAtomI(i, dlcmp, cphi, t1);
         emrecipTorqueAtomI(i, cmp, dlcphi, t2);
         atomic_add((t1[0] + t2[0]) * f, dltrqx, i);
         atomic_add((t1[1] + t2[1]) * f, dltrqy, i);
         atomic_add((t1[2] + t2[2]) * f, dltrqz, i);

         if CONSTEXPR (do_v) {
            real v[6], v1[6], v2[6];
            emrecipVirialAtomI(i, cmp, cphi, v);
            atomic_add(v[0] * f, v[1] * f, v[2] * f, v[3] * f, v[4] * f, v[5] * f, vir_em, ithread);

            emrecipVirialAtomI(i, dlcmp, cphi, v1);
            emrecipVirialAtomI(i, cmp, dlcphi, v2);
            atomic_add((v1[0] + v2[0]) * f, (v1[1] + v2[1]) * f, (v1[2] + v2[2]) * f, (v1[3] + v2[3]) * f,
               (v1[4] + v2[4]) * f, (v1[5] + v2[5]) * f, demvirdl, ithread);
         } // end if (do_v)
      } // end if (do_g)
   }
}

template <class Ver>
static void empoleEwaldRecipDlmdaGeneric_cu()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const PMEUnit pu = epme_unit;
   const PMEUnit dlpu = dlpme_unit;

   cmpToFmp(pu, cmp, fmp);
   cmpToFmp(pu, dlcmp, dlfmp);
   gridMpole(pu, fmp);
   gridMpole(dlpu, dlfmp);
   fftfront(pu);
   fftfront(dlpu);

   if CONSTEXPR (do_v) {
      if (vir_m) {
         pmeConvDlmda(pu, dlpu, vir_m, demvirdl_buf);
         auto size = bufferSize() * VirialBufferTraits::value;
         launch_k1s(g::s0, size, emrecipAddVirial_cu, size, vir_em, vir_m);
      } else {
         pmeConvDlmda(pu, dlpu, vir_em, demvirdl_buf);
      }
   } else {
      pmeConvDlmda(pu, dlpu, nullptr, nullptr);
   }

   fftback(pu);
   fftback(dlpu);
   fphiMpole(pu, fphi);
   fphiMpole(dlpu, dlfphi);
   fphiToCphi(pu, fphi, cphi);
   fphiToCphi(pu, dlfphi, dlcphi);

   auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real f = electric / dielec;

   launch_k1b(g::s0, n, empoleEwaldRecipDlmdaGeneric_cu1<do_e, do_g, do_v>,  //
      n, f, em, demdl_buf, d2emdl2_buf, vir_em, demvirdl_buf,         //
      demx, demy, demz, dfmdlx, dfmdly, dfmdlz,                       //
      trqx, trqy, trqz, dltrqx, dltrqy, dltrqz,                       //
      cmp, fmp, cphi, fphi, dlcmp, dlfmp, dlcphi, dlfphi,             //
      nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
}

void empoleEwaldRecipDlmda_cu(int vers)
{
   if (vers == calc::v0)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V0>();
   else if (vers == calc::v1)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V1>();
   else if (vers == calc::v3)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V3>();
   else if (vers == calc::v4)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V4>();
   else if (vers == calc::v5)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V5>();
   else if (vers == calc::v6)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V6>();
}
}
