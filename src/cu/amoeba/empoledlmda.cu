#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/egvop.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "seq/add.h"
#include "seq/emrecip.h"
#include "seq/launch.h"
#include <cassert>

namespace tinker {
template <class Ver>
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
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS, real deldl, real d2eldl2)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   // dle = dlfmp . fphi feeds both energy lambda derivatives
   constexpr bool do_dle = do_e and (do_dl1 or do_dl2);

   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e, dle, d2le;
      real f1, f2, f3;
      real a1, a2, a3;
      real b1, b2, b3;
      real unused;

      // dlfphi and dlcphi, the potential of the lambda derivative grid, only
      // exist when a second, force or virial lambda derivative is requested.
      emrecipEnergyForceAtomI<do_e, do_g>(i, fmp, fphi, e, f1, f2, f3);
      if CONSTEXPR (do_dle or do_gdl)
         emrecipEnergyForceAtomI<do_dle, do_gdl>(i, dlfmp, fphi, dle, a1, a2, a3);
      if CONSTEXPR (do_gdl)
         emrecipEnergyForceAtomI<false, true>(i, fmp, dlfphi, unused, b1, b2, b3);
      if CONSTEXPR (do_e) {
         atomic_add(0.5f * e * f, em, ithread);
         if CONSTEXPR (do_dl1)
            atomic_add(dle * f * deldl, demdl, ithread);
         if CONSTEXPR (do_dl2) {
            emrecipEnergyForceAtomI<true, false>(i, dlfmp, dlfphi, d2le, unused, unused, unused);
            atomic_add((d2le * deldl * deldl + dle * d2eldl2) * f, d2emdl2, ithread);
         }
      }

      if CONSTEXPR (do_g) {
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;

         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;
         atomic_add(h1 * f, demx, i);
         atomic_add(h2 * f, demy, i);
         atomic_add(h3 * f, demz, i);

         if CONSTEXPR (do_gdl) {
            real dlf1 = (a1 + b1) * nfft1, dlf2 = (a2 + b2) * nfft2, dlf3 = (a3 + b3) * nfft3;
            real dlh1 = recipa.x * dlf1 + recipb.x * dlf2 + recipc.x * dlf3;
            real dlh2 = recipa.y * dlf1 + recipb.y * dlf2 + recipc.y * dlf3;
            real dlh3 = recipa.z * dlf1 + recipb.z * dlf2 + recipc.z * dlf3;
            atomic_add(dlh1 * f * deldl, dfmdlx, i);
            atomic_add(dlh2 * f * deldl, dfmdly, i);
            atomic_add(dlh3 * f * deldl, dfmdlz, i);
         }

         // resolve site torques then increment forces and virial
         real tem[3], t1[3], t2[3];
         emrecipTorqueAtomI(i, cmp, cphi, tem);
         atomic_add(tem[0] * f, trqx, i);
         atomic_add(tem[1] * f, trqy, i);
         atomic_add(tem[2] * f, trqz, i);

         if CONSTEXPR (do_tdl) {
            emrecipTorqueAtomI(i, dlcmp, cphi, t1);
            emrecipTorqueAtomI(i, cmp, dlcphi, t2);
            atomic_add((t1[0] + t2[0]) * f * deldl, dltrqx, i);
            atomic_add((t1[1] + t2[1]) * f * deldl, dltrqy, i);
            atomic_add((t1[2] + t2[2]) * f * deldl, dltrqz, i);
         }

         if CONSTEXPR (do_v) {
            real v[6], v1[6], v2[6];
            emrecipVirialAtomI(i, cmp, cphi, v);
            atomic_add(v[0] * f, v[1] * f, v[2] * f, v[3] * f, v[4] * f, v[5] * f, vir_em, ithread);

            if CONSTEXPR (do_vdl) {
               emrecipVirialAtomI(i, dlcmp, cphi, v1);
               emrecipVirialAtomI(i, cmp, dlcphi, v2);
               atomic_add((v1[0] + v2[0]) * f * deldl, (v1[1] + v2[1]) * f * deldl,
                  (v1[2] + v2[2]) * f * deldl, (v1[3] + v2[3]) * f * deldl, (v1[4] + v2[4]) * f * deldl,
                  (v1[5] + v2[5]) * f * deldl, demvirdl, ithread);
            }
         } // end if (do_v)
      } // end if (do_g)
   }
}

// With reuse_pot, fmp, fphi and cphi are the ones induce() just built for the
// direct field from this same cmp, and the convolution virial is already in
// vir_m, so only dlfmp and the per-atom work are left.
template <class Ver>
static void empoleEwaldRecipDlmdaGeneric_cu(bool reuse_pot)
{
   constexpr bool do_v = Ver::v;
   // dE/dL = dlfmp . fphi needs only the ordinary grid; the lambda derivative
   // grid is built for the second, force and virial lambda derivatives alone
   // (empole4.f builds lqgrid only under use_d2lmda).
   constexpr bool need_dl = Ver::e_dlmda2 or Ver::g_dlmda or Ver::v_dlmda;
   // dlfmp is read by dE/dL and by everything the second grid serves
   constexpr bool need_dlfmp = Ver::e_dlmda1 or need_dl;

   const PMEUnit pu = epme_unit;
   const PMEUnit dlpu = dlpme_unit;

   if CONSTEXPR (need_dlfmp)
      cmpToFmp(pu, dlcmp, dlfmp);

   if (reuse_pot) {
      // the lambda derivative grid would need a convolution of its own
      assert(not need_dl);
      if CONSTEXPR (do_v) {
         if (vir_m) {
            auto size = bufferSize() * VirialBufferTraits::value;
            sumVirialBuffer(size, vir_em, vir_m);
         }
      }
   } else {
      cmpToFmp(pu, cmp, fmp);
      gridMpole(pu, fmp);
      fftfront(pu);
      if CONSTEXPR (need_dl) {
         gridMpole(dlpu, dlfmp);
         fftfront(dlpu);
      }

      VirialBuffer conv_vir = nullptr;
      if CONSTEXPR (do_v)
         conv_vir = vir_m ? vir_m : vir_em;
      if CONSTEXPR (need_dl) {
         pmeConvDlmda(pu, dlpu, conv_vir, conv_vir ? dvirdl_buf : nullptr, deldlmda);
      } else {
         if (conv_vir)
            pmeConv(pu, conv_vir);
         else
            pmeConv(pu);
      }
      if CONSTEXPR (do_v) {
         if (vir_m) {
            auto size = bufferSize() * VirialBufferTraits::value;
            sumVirialBuffer(size, vir_em, vir_m);
         }
      }

      fftback(pu);
      fphiMpole(pu, fphi);
      fphiToCphi(pu, fphi, cphi);
      if CONSTEXPR (need_dl) {
         fftback(dlpu);
         fphiMpole(dlpu, dlfphi);
         fphiToCphi(pu, dlfphi, dlcphi);
      }
   }

   auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real f = electric / dielec;

   launch_k1b(g::s0, n, empoleEwaldRecipDlmdaGeneric_cu1<Ver>,         //
      n, f, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf,           //
      demx, demy, demz, dfdlx, dfdly, dfdlz,                 //
      trqx, trqy, trqz, dltrqx, dltrqy, dltrqz,                       //
      cmp, fmp, cphi, fphi, dlcmp, dlfmp, dlcphi, dlfphi,             //
      nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS, deldlmda, d2eldlmda2);
}

void empoleEwaldRecipDlmda_cu(int vers, bool reuse_pot)
{
   if (vers == calc::v0)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V0>(reuse_pot);
   else if (vers == calc::v1)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V1>(reuse_pot);
   else if (vers == calc::v3)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V3>(reuse_pot);
   else if (vers == calc::v4)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V4>(reuse_pot);
   else if (vers == calc::v5)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V5>(reuse_pot);
   else if (vers == calc::v6)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V6>(reuse_pot);
   else if (vers == calc::v7)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V7>(reuse_pot);
   else if (vers == calc::v8)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V8>(reuse_pot);
   else if (vers == calc::v9)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V9>(reuse_pot);
   else if (vers == calc::v10)
      empoleEwaldRecipDlmdaGeneric_cu<calc::V10>(reuse_pot);
}
}
