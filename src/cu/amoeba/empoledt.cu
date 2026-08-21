#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/egvop.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/emrecip.h"
#include "seq/emselfamoeba.h"
#include "seq/launch.h"
#include "seq/pair_mpole.h"
#include "seq/triangle.h"
#include <tinker/detail/extfld.hh>

// One fused pass over both dual topology endpoints for the permanent
// multipoles. The endpoints differ only by which interactions they include --
// rotpoleState used to build them by zeroing multipoles, and there is no
// softcore -- so a pair's energy is the same in both. The kernel therefore
// evaluates each pair once at full coupling and splits the result three ways,
// with weights chosen by the groups of the two atoms:
//
//     E = a0*E(state 0) + a1*E(state 1),  a0 = 1-w, a1 = w
//
// and likewise dE/dlambda with b, d2E/dlambda2 with c. See DtCoef.
//
// A term over a single atom -- the Ewald self energy, the external field --
// takes the diagonal cell (g,g), which is set in an endpoint's mask exactly
// when that endpoint contains the atom's group. Every atom therefore
// contributes once per endpoint, as it must.

namespace tinker {
#include "empoledt_cu1.cc"

__device__
inline void dtCellWeights(unsigned in0bits, unsigned in1bits, unsigned cntbits, int cell, //
   real a0, real a1, real b0, real b1, real c0, real c1,                                  //
   real& wa, real& wb, real& wc, bool& cnt)
{
   bool in0 = (in0bits >> cell) & 1;
   bool in1 = (in1bits >> cell) & 1;
   cnt = (cntbits >> cell) & 1;
   wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
   wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
   wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
}

template <class Ver>
__global__
static void empoleSelfDt_cu(CountBuffer restrict nem, EnergyBuffer restrict em, EnergyBuffer restrict demdl,
   EnergyBuffer restrict d2emdl2, const real (*restrict rpole)[10], const int* restrict grp, int n, real f,
   real aewald, unsigned in0bits, unsigned in1bits, unsigned cntbits, real a0, real a1, real b0, real b1,
   real c0, real c1, int nself)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;

   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

   for (int i = ITHREAD; i < n; i += STRIDE) {
      int offset = ITHREAD;
      int g = grp[i];
      real wa, wb, wc;
      bool cnt;
      dtCellWeights(in0bits, in1bits, cntbits, 3 * g + g, a0, a1, b0, b1, c0, c1, wa, wb, wc, cnt);

      real e = empoleSelfEnergyAtomI(i, rpole, fterm, aewald_sq_2);
      atomic_add(wa * e, em, offset);
      if CONSTEXPR (do_dl1)
         atomic_add(wb * e, demdl, offset);
      if CONSTEXPR (do_dl2)
         atomic_add(wc * e, d2emdl2, offset);
      if CONSTEXPR (do_a) {
         atomic_add(nself, nem, offset);
      }
   }
}

template <class Ver>
static void empoledt_cu(const DtCoef& c, const int* grp, int nself)
{
   constexpr bool do_e = Ver::e;

   const auto& st = *mspatial_v2_unit;
   const real off = useEwald() ? switchOff(Switch::EWALD) : switchOff(Switch::MPOLE);
   const real f = electric / dielec;
   real aewald = 0;

   if (useEwald()) {
      aewald = epme_unit->aewald;
      if CONSTEXPR (do_e) {
         launch_k1b(g::s0, n, empoleSelfDt_cu<Ver>, //
            nem, em, demdl_buf, d2emdl2_buf, rpole, grp, n, f, aewald, c.in0bits, c.in1bits, c.cntbits, c.a0,
            c.a1, c.b0, c.b1, c.c0, c.c1, nself);
      }
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   if (useEwald()) {
      empoledt_cu1<Ver, EWALD><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, demdl_buf,
         d2emdl2_buf, vir_em, dvirdl_buf, demx, demy, demz, dfsumdlx, dfsumdly, dfsumdlz, off, st.si1.bit0,
         nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak,
         st.iak, st.lst, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, grp, f, aewald, c.in0bits, c.in1bits,
         c.cntbits, c.a0, c.a1, c.b0, c.b1, c.c0, c.c1);
   } else {
      empoledt_cu1<Ver, NON_EWALD><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, demdl_buf,
         d2emdl2_buf, vir_em, dvirdl_buf, demx, demy, demz, dfsumdlx, dfsumdly, dfsumdlz, off, st.si1.bit0,
         nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak,
         st.iak, st.lst, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, grp, f, aewald, c.in0bits, c.in1bits,
         c.cntbits, c.a0, c.a1, c.b0, c.b1, c.c0, c.c1);
   }
}

void empoleDt_cu(int vers, const DtCoef& coef, int nself)
{
   // The relative schedule labels atoms by ligand; the absolute one only knows
   // mutated from not.
   const int* grp = use_rel ? rdt_group : mut;

   if (vers == calc::v0)
      empoledt_cu<calc::V0>(coef, grp, nself);
   else if (vers == calc::v1)
      empoledt_cu<calc::V1>(coef, grp, nself);
   else if (vers == calc::v3)
      empoledt_cu<calc::V3>(coef, grp, nself);
   else if (vers == calc::v4)
      empoledt_cu<calc::V4>(coef, grp, nself);
   else if (vers == calc::v5)
      empoledt_cu<calc::V5>(coef, grp, nself);
   else if (vers == calc::v6)
      empoledt_cu<calc::V6>(coef, grp, nself);
   else if (vers == calc::v7)
      empoledt_cu<calc::V7>(coef, grp, nself);
   else if (vers == calc::v8)
      empoledt_cu<calc::V8>(coef, grp, nself);
   else if (vers == calc::v9)
      empoledt_cu<calc::V9>(coef, grp, nself);
   else if (vers == calc::v10)
      empoledt_cu<calc::V10>(coef, grp, nself);
}
}

namespace tinker {
template <class Ver>
__global__
static void exfieldDipoleDt_cu1(CountBuffer restrict nem, EnergyBuffer restrict em, EnergyBuffer restrict demdl,
   EnergyBuffer restrict d2emdl2, VirialBuffer restrict vir_em, VirialBuffer restrict demvirdl,             //
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz,                            //
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz,                      //
   real* restrict trqx, real* restrict trqy, real* restrict trqz,                                           //
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz,                                     //
   int n, real f, real ef1, real ef2, real ef3, const real (*restrict rpole)[10], const int* restrict grp,
   const real* restrict x, const real* restrict y, const real* restrict z, unsigned in0bits, unsigned in1bits,
   unsigned cntbits, real a0, real a1, real b0, real b1, real c0, real c1)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii];
      real ci = rpole[ii][0], dix = rpole[ii][1], diy = rpole[ii][2], diz = rpole[ii][3];

      int g = grp[ii];
      real wa, wb, wc;
      bool cnt;
      dtCellWeights(in0bits, in1bits, cntbits, 3 * g + g, a0, a1, b0, b1, c0, c1, wa, wb, wc, cnt);

      if CONSTEXPR (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * (ci * phi + dix * ef1 + diy * ef2 + diz * ef3);
         atomic_add(wa * e, em, ithread);
         if CONSTEXPR (do_dl1)
            atomic_add(wb * e, demdl, ithread);
         if CONSTEXPR (do_dl2)
            atomic_add(wc * e, d2emdl2, ithread);
         if CONSTEXPR (do_a) {
            if (cnt and e != 0)
               atomic_add(1, nem, ithread);
         }
      }
      if CONSTEXPR (do_g) {
         // unscaled torque due to the dipole
         real tx = f * (diy * ef3 - diz * ef2);
         real ty = f * (diz * ef1 - dix * ef3);
         real tz = f * (dix * ef2 - diy * ef1);
         atomic_add(wa * tx, trqx, ii);
         atomic_add(wa * ty, trqy, ii);
         atomic_add(wa * tz, trqz, ii);
         // unscaled gradient and virial due to the monopole
         real frx = -f * ef1 * ci;
         real fry = -f * ef2 * ci;
         real frz = -f * ef3 * ci;
         atomic_add(wa * frx, demx, ii);
         atomic_add(wa * fry, demy, ii);
         atomic_add(wa * frz, demz, ii);
         if CONSTEXPR (do_tdl) {
            atomic_add(wb * tx, dltrqx, ii);
            atomic_add(wb * ty, dltrqy, ii);
            atomic_add(wb * tz, dltrqz, ii);
         }
         if CONSTEXPR (do_gdl) {
            atomic_add(wb * frx, dfmdlx, ii);
            atomic_add(wb * fry, dfmdly, ii);
            atomic_add(wb * frz, dfmdlz, ii);
         }
         if CONSTEXPR (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(wa * vxx, wa * vxy, wa * vxz, wa * vyy, wa * vyz, wa * vzz, vir_em, ithread);
            if CONSTEXPR (do_vdl) {
               atomic_add(wb * vxx, wb * vxy, wb * vxz, wb * vyy, wb * vyz, wb * vzz, demvirdl, ithread);
            }
         }
      }
   }
}

template <class Ver>
static void exfielddt_cu(const DtCoef& c, const int* grp)
{
   real f = electric / dielec;
   real ef1 = extfld::texfld[0], ef2 = extfld::texfld[1], ef3 = extfld::texfld[2];
   launch_k1b(g::s0, n, exfieldDipoleDt_cu1<Ver>, nem, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf, demx,
      demy, demz, dfsumdlx, dfsumdly, dfsumdlz, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, n, f, ef1, ef2, ef3, rpole,
      grp, x, y, z, c.in0bits, c.in1bits, c.cntbits, c.a0, c.a1, c.b0, c.b1, c.c0, c.c1);
}

void exfieldDipoleDt_cu(int vers, const DtCoef& coef)
{
   const int* grp = use_rel ? rdt_group : mut;

   if (vers == calc::v0)
      exfielddt_cu<calc::V0>(coef, grp);
   else if (vers == calc::v1)
      exfielddt_cu<calc::V1>(coef, grp);
   else if (vers == calc::v3)
      exfielddt_cu<calc::V3>(coef, grp);
   else if (vers == calc::v4)
      exfielddt_cu<calc::V4>(coef, grp);
   else if (vers == calc::v5)
      exfielddt_cu<calc::V5>(coef, grp);
   else if (vers == calc::v6)
      exfielddt_cu<calc::V6>(coef, grp);
   else if (vers == calc::v7)
      exfielddt_cu<calc::V7>(coef, grp);
   else if (vers == calc::v8)
      exfielddt_cu<calc::V8>(coef, grp);
   else if (vers == calc::v9)
      exfielddt_cu<calc::V9>(coef, grp);
   else if (vers == calc::v10)
      exfielddt_cu<calc::V10>(coef, grp);
}
}

namespace tinker {
// One subsystem's reciprocal space, weighted into all three channels. Unlike
// the analytic-scaling kernel there is no cross term to pick up: the multipoles
// do not depend on lambda here, only the weight does, so every channel is the
// same bilinear form with a different scalar in front -- and every one of them
// carries the same one half.
template <class Ver>
__global__
static void empoleEwaldRecipDt_cu1(int n, real f,                                                    //
   EnergyBuffer restrict em, EnergyBuffer restrict demdl, EnergyBuffer restrict d2emdl2,             //
   VirialBuffer restrict vir_em, VirialBuffer restrict demvirdl,                                     //
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz,                     //
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz,               //
   real* restrict trqx, real* restrict trqy, real* restrict trqz,                                    //
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz,                              //
   const real (*restrict cmp)[10], const real (*restrict fmp)[10],                                   //
   const real (*restrict cphi)[10], const real (*restrict fphi)[20],                                 //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS, real wa, real wb, real wc)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;

   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e, f1, f2, f3;
      emrecipEnergyForceAtomI<do_e, do_g>(i, fmp, fphi, e, f1, f2, f3);

      if CONSTEXPR (do_e) {
         real half = 0.5f * e * f;
         atomic_add(wa * half, em, ithread);
         if CONSTEXPR (do_dl1)
            atomic_add(wb * half, demdl, ithread);
         if CONSTEXPR (do_dl2)
            atomic_add(wc * half, d2emdl2, ithread);
      }

      if CONSTEXPR (do_g) {
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;

         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;
         atomic_add(wa * h1 * f, demx, i);
         atomic_add(wa * h2 * f, demy, i);
         atomic_add(wa * h3 * f, demz, i);
         if CONSTEXPR (do_gdl) {
            atomic_add(wb * h1 * f, dfmdlx, i);
            atomic_add(wb * h2 * f, dfmdly, i);
            atomic_add(wb * h3 * f, dfmdlz, i);
         }

         // resolve site torques then increment forces and virial
         real tem[3];
         emrecipTorqueAtomI(i, cmp, cphi, tem);
         atomic_add(wa * tem[0] * f, trqx, i);
         atomic_add(wa * tem[1] * f, trqy, i);
         atomic_add(wa * tem[2] * f, trqz, i);
         if CONSTEXPR (do_tdl) {
            atomic_add(wb * tem[0] * f, dltrqx, i);
            atomic_add(wb * tem[1] * f, dltrqy, i);
            atomic_add(wb * tem[2] * f, dltrqz, i);
         }

         if CONSTEXPR (do_v) {
            real v[6];
            emrecipVirialAtomI(i, cmp, cphi, v);
            atomic_add(wa * v[0] * f, wa * v[1] * f, wa * v[2] * f, wa * v[3] * f, wa * v[4] * f, wa * v[5] * f,
               vir_em, ithread);
            if CONSTEXPR (do_vdl) {
               atomic_add(wb * v[0] * f, wb * v[1] * f, wb * v[2] * f, wb * v[3] * f, wb * v[4] * f,
                  wb * v[5] * f, demvirdl, ithread);
            }
         }
      }
   }
}

template <class Ver>
static void empolerecipdt_cu(RdtMask mask, const int* grp, real wa, real wb, real wc)
{
   constexpr bool do_v = Ver::v;

   const PMEUnit pu = epme_unit;

   rpoleToCmpState(mask, grp);
   cmpToFmp(pu, cmp, fmp);
   gridMpole(pu, fmp);
   fftfront(pu);
   if CONSTEXPR (do_v) {
      pmeConvDt(pu, vir_em, wa, dvirdl_buf, wb);
   } else {
      pmeConvDt(pu, nullptr, wa, nullptr, wb);
   }
   fftback(pu);
   fphiMpole(pu, fphi);
   fphiToCphi(pu, fphi, cphi);

   auto& st = *pu;
   const real f = electric / dielec;

   launch_k1b(g::s0, n, empoleEwaldRecipDt_cu1<Ver>,                    //
      n, f, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf,             //
      demx, demy, demz, dfsumdlx, dfsumdly, dfsumdlz,                   //
      trqx, trqy, trqz, dltrqx, dltrqy, dltrqz,                         //
      cmp, fmp, cphi, fphi,                                             //
      st.nfft1, st.nfft2, st.nfft3, TINKER_IMAGE_ARGS, wa, wb, wc);
}

void empoleEwaldRecipDt_cu(int vers, RdtMask mask, real wa, real wb, real wc)
{
   const int* grp = use_rel ? rdt_group : mut;

   if (vers == calc::v0)
      empolerecipdt_cu<calc::V0>(mask, grp, wa, wb, wc);
   else if (vers == calc::v1)
      empolerecipdt_cu<calc::V1>(mask, grp, wa, wb, wc);
   else if (vers == calc::v3)
      empolerecipdt_cu<calc::V3>(mask, grp, wa, wb, wc);
   else if (vers == calc::v4)
      empolerecipdt_cu<calc::V4>(mask, grp, wa, wb, wc);
   else if (vers == calc::v5)
      empolerecipdt_cu<calc::V5>(mask, grp, wa, wb, wc);
   else if (vers == calc::v6)
      empolerecipdt_cu<calc::V6>(mask, grp, wa, wb, wc);
   else if (vers == calc::v7)
      empolerecipdt_cu<calc::V7>(mask, grp, wa, wb, wc);
   else if (vers == calc::v8)
      empolerecipdt_cu<calc::V8>(mask, grp, wa, wb, wc);
   else if (vers == calc::v9)
      empolerecipdt_cu<calc::V9>(mask, grp, wa, wb, wc);
   else if (vers == calc::v10)
      empolerecipdt_cu<calc::V10>(mask, grp, wa, wb, wc);
}
}
