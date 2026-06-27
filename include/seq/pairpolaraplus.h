#pragma once
#include "seq/damp.h"
#include "seq/dampaplus.h"
#include "seq/pair_polar.h"

namespace tinker {
#pragma acc routine seq
template <bool do_e, bool do_g, class ETYP, int CFLX>
SEQ_CUDA
void pair_polar_aplus(real r2,
                      real xr,
                      real yr,
                      real zr,
                      real dscale,
                      real uscale,
                      real ci,
                      real dix,
                      real diy,
                      real diz,
                      real pdi,
                      real pti,
                      real ddi,
                      real qixx,
                      real qixy,
                      real qixz,
                      real qiyy,
                      real qiyz,
                      real qizz,
                      real uix,
                      real uiy,
                      real uiz,
                      real ck,
                      real dkx,
                      real dky,
                      real dkz,
                      real pdk,
                      real ptk,
                      real ddk,
                      real qkxx,
                      real qkxy,
                      real qkxz,
                      real qkyy,
                      real qkyz,
                      real qkzz,
                      real ukx,
                      real uky,
                      real ukz,
                      real f,
                      real aewald,
                      real& restrict e,
                      real& restrict poti,
                      real& restrict potk,
                      PairPolarGrad& restrict pgrad)
{
   real dir = dix * xr + diy * yr + diz * zr;
   real qix = qixx * xr + qixy * yr + qixz * zr;
   real qiy = qixy * xr + qiyy * yr + qiyz * zr;
   real qiz = qixz * xr + qiyz * yr + qizz * zr;
   real qir = qix * xr + qiy * yr + qiz * zr;
   real dkr = dkx * xr + dky * yr + dkz * zr;
   real qkx = qkxx * xr + qkxy * yr + qkxz * zr;
   real qky = qkxy * xr + qkyy * yr + qkyz * zr;
   real qkz = qkxz * xr + qkyz * yr + qkzz * zr;
   real qkr = qkx * xr + qky * yr + qkz * zr;
   real uir = uix * xr + uiy * yr + uiz * zr;
   real ukr = ukx * xr + uky * yr + ukz * zr;

   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   MAYBE_UNUSED real rr9;
   if CONSTEXPR (do_g)
      rr9 = 7 * rr7 * rr2;
   real bn[5];
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (!do_g)
         damp_ewald<4>(bn, r, invr1, rr2, aewald);
      else
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = rr3;
      bn[2] = rr5;
      bn[3] = rr7;
      if CONSTEXPR (do_g)
         bn[4] = rr9;
   }

   // if use_dirdamp
   real ex3, ex5, ex7;
   MAYBE_UNUSED real rc31, rc32, rc33, rc51, rc52, rc53, rc71, rc72, rc73;
   if CONSTEXPR (!do_g) {
      damp_aplus3(r, pdi, ddi, pdk, ddk, //
                  ex3, ex5, ex7);
      ex3 = 1 - ex3;
      ex5 = 1 - ex5;
      ex7 = 1 - ex7;
   } else {
      damp_aplus3g(          //
         r, rr2, xr, yr, zr, //
         pdi, ddi, pdk, ddk, //
         ex3, ex5, ex7,      //
         rc31, rc32, rc33,   //
         rc51, rc52, rc53,   //
         rc71, rc72, rc73);
      rc31 *= rr3;
      rc32 *= rr3;
      rc33 *= rr3;
      rc51 *= rr5;
      rc52 *= rr5;
      rc53 *= rr5;
      rc71 *= rr7;
      rc72 *= rr7;
      rc73 *= rr7;
   }
   // end if use_dirdamp

   real sr3 = bn[1] - ex3 * rr3;
   real sr5 = bn[2] - ex5 * rr5;
   real sr7 = bn[3] - ex7 * rr7;

   if CONSTEXPR (do_e) {
      real diu = dix * ukx + diy * uky + diz * ukz;
      real qiu = qix * ukx + qiy * uky + qiz * ukz;
      real dku = dkx * uix + dky * uiy + dkz * uiz;
      real qku = qkx * uix + qky * uiy + qkz * uiz;
      real term1 = ck * uir - ci * ukr + diu + dku;
      real term2 = 2 * (qiu - qku) - uir * dkr - dir * ukr;
      real term3 = uir * qkr - ukr * qir;
      e = dscale * f * (term1 * sr3 + term2 * sr5 + term3 * sr7);
   }

   if CONSTEXPR (do_g) {
      real uir = uix * xr + uiy * yr + uiz * zr;
      real ukr = ukx * xr + uky * yr + ukz * zr;
      if CONSTEXPR (CFLX) {
         poti = -2.0 * ukr * f * dscale * sr3;
         potk = 2.0 * uir * f * dscale * sr3;
      }

      // get the induced dipole field used for dipole torques

      real tuir, tukr;

      real tix3 = 2.0 * dscale * sr3 * ukx;
      real tiy3 = 2.0 * dscale * sr3 * uky;
      real tiz3 = 2.0 * dscale * sr3 * ukz;
      real tkx3 = 2.0 * dscale * sr3 * uix;
      real tky3 = 2.0 * dscale * sr3 * uiy;
      real tkz3 = 2.0 * dscale * sr3 * uiz;
      tuir = -2.0 * dscale * sr5 * ukr;
      tukr = -2.0 * dscale * sr5 * uir;

      pgrad.ufldi[0] = f * (tix3 + xr * tuir);
      pgrad.ufldi[1] = f * (tiy3 + yr * tuir);
      pgrad.ufldi[2] = f * (tiz3 + zr * tuir);
      pgrad.ufldk[0] = f * (tkx3 + xr * tukr);
      pgrad.ufldk[1] = f * (tky3 + yr * tukr);
      pgrad.ufldk[2] = f * (tkz3 + zr * tukr);

      // get induced dipole field gradient used for quadrupole torques

      real tix5 = 4 * (dscale * sr5 * ukx);
      real tiy5 = 4 * (dscale * sr5 * uky);
      real tiz5 = 4 * (dscale * sr5 * ukz);
      real tkx5 = 4 * (dscale * sr5 * uix);
      real tky5 = 4 * (dscale * sr5 * uiy);
      real tkz5 = 4 * (dscale * sr5 * uiz);
      tuir = -2.0 * dscale * sr7 * ukr;
      tukr = -2.0 * dscale * sr7 * uir;

      pgrad.dufldi[0] = f * (xr * tix5 + xr * xr * tuir);
      pgrad.dufldi[1] = f * (xr * tiy5 + yr * tix5 + 2 * xr * yr * tuir);
      pgrad.dufldi[2] = f * (yr * tiy5 + yr * yr * tuir);
      pgrad.dufldi[3] = f * (xr * tiz5 + zr * tix5 + 2 * xr * zr * tuir);
      pgrad.dufldi[4] = f * (yr * tiz5 + zr * tiy5 + 2 * yr * zr * tuir);
      pgrad.dufldi[5] = f * (zr * tiz5 + zr * zr * tuir);
      pgrad.dufldk[0] = f * (-xr * tkx5 - xr * xr * tukr);
      pgrad.dufldk[1] = f * (-xr * tky5 - yr * tkx5 - 2 * xr * yr * tukr);
      pgrad.dufldk[2] = f * (-yr * tky5 - yr * yr * tukr);
      pgrad.dufldk[3] = f * (-xr * tkz5 - zr * tkx5 - 2 * xr * zr * tukr);
      pgrad.dufldk[4] = f * (-yr * tkz5 - zr * tky5 - 2 * yr * zr * tukr);
      pgrad.dufldk[5] = f * (-zr * tkz5 - zr * zr * tukr);

      // get the field gradient for direct polarization force

      real term1, term2, term3, term4, term5, term6, term7;

      term1 = bn[2] - ex3 * rr5;
      term2 = bn[3] - ex5 * rr7;
      term3 = -sr3 + term1 * xr * xr - xr * rc31;
      term4 = rc31 - term1 * xr - sr5 * xr;
      term5 = term2 * xr * xr - sr5 - xr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * xr - bn[3] - xr * rc71;
      term7 = rc51 - 2 * bn[3] * xr + (ex5 + 1.5f * ex7) * rr7 * xr;
      real tixx = ci * term3 + dix * term4 + dir * term5 + 2 * sr5 * qixx + (qiy * yr + qiz * zr) * ex7 * rr7 + 2 * qix * term7 + qir * term6;
      real tkxx = ck * term3 - dkx * term4 - dkr * term5 + 2 * sr5 * qkxx + (qky * yr + qkz * zr) * ex7 * rr7 + 2 * qkx * term7 + qkr * term6;

      term3 = -sr3 + term1 * yr * yr - yr * rc32;
      term4 = rc32 - term1 * yr - sr5 * yr;
      term5 = term2 * yr * yr - sr5 - yr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * yr - bn[3] - yr * rc72;
      term7 = rc52 - 2 * bn[3] * yr + (ex5 + 1.5f * ex7) * rr7 * yr;
      real tiyy = ci * term3 + diy * term4 + dir * term5 + 2 * sr5 * qiyy + (qix * xr + qiz * zr) * ex7 * rr7 + 2 * qiy * term7 + qir * term6;
      real tkyy = ck * term3 - dky * term4 - dkr * term5 + 2 * sr5 * qkyy + (qkx * xr + qkz * zr) * ex7 * rr7 + 2 * qky * term7 + qkr * term6;

      term3 = -sr3 + term1 * zr * zr - zr * rc33;
      term4 = rc33 - term1 * zr - sr5 * zr;
      term5 = term2 * zr * zr - sr5 - zr * rc53;
      term6 = (bn[4] - ex7 * rr9) * zr * zr - bn[3] - zr * rc73;
      term7 = rc53 - 2 * bn[3] * zr + (ex5 + 1.5f * ex7) * rr7 * zr;
      real tizz = ci * term3 + diz * term4 + dir * term5 + 2 * sr5 * qizz + (qix * xr + qiy * yr) * ex7 * rr7 + 2 * qiz * term7 + qir * term6;
      real tkzz = ck * term3 - dkz * term4 - dkr * term5 + 2 * sr5 * qkzz + (qkx * xr + qky * yr) * ex7 * rr7 + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * xr * yr - yr * rc31;
      term4 = rc31 - term1 * xr;
      term5 = term2 * xr * yr - yr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * yr - yr * rc71;
      term7 = rc51 - term2 * xr;
      real tixy = ci * term3 - sr5 * dix * yr + diy * term4 + dir * term5 + 2 * sr5 * qixy - 2 * sr7 * yr * qix + 2 * qiy * term7 + qir * term6;
      real tkxy = ck * term3 + sr5 * dkx * yr - dky * term4 - dkr * term5 + 2 * sr5 * qkxy - 2 * sr7 * yr * qkx + 2 * qky * term7 + qkr * term6;

      term3 = term1 * xr * zr - zr * rc31;
      term5 = term2 * xr * zr - zr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * zr - zr * rc71;
      real tixz = ci * term3 - sr5 * dix * zr + diz * term4 + dir * term5 + 2 * sr5 * qixz - 2 * sr7 * zr * qix + 2 * qiz * term7 + qir * term6;
      real tkxz = ck * term3 + sr5 * dkx * zr - dkz * term4 - dkr * term5 + 2 * sr5 * qkxz - 2 * sr7 * zr * qkx + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * yr * zr - zr * rc32;
      term4 = rc32 - term1 * yr;
      term5 = term2 * yr * zr - zr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * zr - zr * rc72;
      term7 = rc52 - term2 * yr;
      real tiyz = ci * term3 - sr5 * diy * zr + diz * term4 + dir * term5 + 2 * sr5 * qiyz - 2 * sr7 * zr * qiy + 2 * qiz * term7 + qir * term6;
      real tkyz = ck * term3 + sr5 * dky * zr - dkz * term4 - dkr * term5 + 2 * sr5 * qkyz - 2 * sr7 * zr * qky + 2 * qkz * term7 + qkr * term6;

      // get the dEd/dR terms for Thole direct polarization force

      real depx, depy, depz;

      depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy - tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy - tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy - tkzz * uiz;
      if CONSTEXPR (eq<ETYP, EWALD>()) {
         pgrad.frcx = -2 * depx;
         pgrad.frcy = -2 * depy;
         pgrad.frcz = -2 * depz;
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         pgrad.frcx = -2 * depx * dscale;
         pgrad.frcy = -2 * depy * dscale;
         pgrad.frcz = -2 * depz * dscale;
      }

      // get the dtau/dr terms used for mutual polarization force
      damp_thole3g(          //
         r, rr2, xr, yr, zr, //
         pdi, pti, pdk, ptk, //
         ex3, ex5, ex7,      //
         rc31, rc32, rc33,   //
         rc51, rc52, rc53,   //
         rc71, rc72, rc73);
      rc31 *= rr3;
      rc32 *= rr3;
      rc33 *= rr3;
      rc51 *= rr5;
      rc52 *= rr5;
      rc53 *= rr5;
      sr3 = bn[1] - ex3 * rr3;
      sr5 = bn[2] - ex5 * rr5;

      term1 = bn[2] - ex3 * rr5;
      term2 = bn[3] - ex5 * rr7;
      term3 = sr5 + term1;

      term5 = -xr * term3 + rc31;
      term6 = -sr5 + xr * xr * term2 - xr * rc51;
      tixx = uix * term5 + uir * term6;
      tkxx = ukx * term5 + ukr * term6;

      term5 = -yr * term3 + rc32;
      term6 = -sr5 + yr * yr * term2 - yr * rc52;
      tiyy = uiy * term5 + uir * term6;
      tkyy = uky * term5 + ukr * term6;

      term5 = -zr * term3 + rc33;
      term6 = -sr5 + zr * zr * term2 - zr * rc53;
      tizz = uiz * term5 + uir * term6;
      tkzz = ukz * term5 + ukr * term6;

      term4 = -sr5 * yr;
      term5 = -xr * term1 + rc31;
      term6 = xr * yr * term2 - yr * rc51;
      tixy = uix * term4 + uiy * term5 + uir * term6;
      tkxy = ukx * term4 + uky * term5 + ukr * term6;

      term4 = -sr5 * zr;
      term6 = xr * zr * term2 - zr * rc51;
      tixz = uix * term4 + uiz * term5 + uir * term6;
      tkxz = ukx * term4 + ukz * term5 + ukr * term6;

      term5 = -yr * term1 + rc32;
      term6 = yr * zr * term2 - zr * rc52;
      tiyz = uiy * term4 + uiz * term5 + uir * term6;
      tkyz = uky * term4 + ukz * term5 + ukr * term6;

      depx = tixx * ukx + tixy * uky + tixz * ukz + tkxx * uix + tkxy * uiy + tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz + tkxy * uix + tkyy * uiy + tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz + tkxz * uix + tkyz * uiy + tkzz * uiz;
      if CONSTEXPR (eq<ETYP, EWALD>()) {
         pgrad.frcx -= depx;
         pgrad.frcy -= depy;
         pgrad.frcz -= depz;
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         pgrad.frcx -= uscale * depx;
         pgrad.frcy -= uscale * depy;
         pgrad.frcz -= uscale * depz;
      }

      pgrad.frcx *= f;
      pgrad.frcy *= f;
      pgrad.frcz *= f;
   }
}

#pragma acc routine seq
template <class Ver, class ETYP, int CFLX>
SEQ_CUDA
void pair_polar_aplus_v2(real r2,
                         real xr,
                         real yr,
                         real zr,
                         real dscale,
                         real uscale,
                         real ci,
                         real dix,
                         real diy,
                         real diz,
                         real qixx,
                         real qixy,
                         real qixz,
                         real qiyy,
                         real qiyz,
                         real qizz,
                         real uix,
                         real uiy,
                         real uiz,
                         real pdi,
                         real pti,
                         real ddi,
                         real ck,
                         real dkx,
                         real dky,
                         real dkz,
                         real qkxx,
                         real qkxy,
                         real qkxz,
                         real qkyy,
                         real qkyz,
                         real qkzz,
                         real ukx,
                         real uky,
                         real ukz,
                         real pdk,
                         real ptk,
                         real ddk,
                         real f,
                         real aewald,
                         real& restrict frcxi,
                         real& restrict frcyi,
                         real& restrict frczi,
                         real& restrict frcxk,
                         real& restrict frcyk,
                         real& restrict frczk, //
                         real& restrict uf0i,
                         real& restrict uf1i,
                         real& restrict uf2i,
                         real& restrict uf0k,
                         real& restrict uf1k,
                         real& restrict uf2k,
                         real& restrict duf0i,
                         real& restrict duf1i,
                         real& restrict duf2i,
                         real& restrict duf3i,
                         real& restrict duf4i,
                         real& restrict duf5i,
                         real& restrict duf0k,
                         real& restrict duf1k,
                         real& restrict duf2k,
                         real& restrict duf3k,
                         real& restrict duf4k,
                         real& restrict duf5k,
                         real& restrict e,
                         real& restrict vxx,
                         real& restrict vxy,
                         real& restrict vxz,
                         real& restrict vyy,
                         real& restrict vyz,
                         real& restrict vzz,
                         real& restrict poti,
                         real& restrict potk)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      dscale = 1;
      // pscale = 1;
      uscale = 1;
   }

   real dir = dix * xr + diy * yr + diz * zr;
   real qix = qixx * xr + qixy * yr + qixz * zr;
   real qiy = qixy * xr + qiyy * yr + qiyz * zr;
   real qiz = qixz * xr + qiyz * yr + qizz * zr;
   real qir = qix * xr + qiy * yr + qiz * zr;
   real dkr = dkx * xr + dky * yr + dkz * zr;
   real qkx = qkxx * xr + qkxy * yr + qkxz * zr;
   real qky = qkxy * xr + qkyy * yr + qkyz * zr;
   real qkz = qkxz * xr + qkyz * yr + qkzz * zr;
   real qkr = qkx * xr + qky * yr + qkz * zr;
   real uir = uix * xr + uiy * yr + uiz * zr;
   real ukr = ukx * xr + uky * yr + ukz * zr;

   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   MAYBE_UNUSED real rr9;
   if CONSTEXPR (do_g)
      rr9 = 7 * rr7 * rr2;
   real bn[5];
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (!do_g)
         damp_ewald<4>(bn, r, invr1, rr2, aewald);
      else
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = rr3;
      bn[2] = rr5;
      bn[3] = rr7;
      if CONSTEXPR (do_g)
         bn[4] = rr9;
   }

   // if use_dirdamp
   real ex3, ex5, ex7;
   MAYBE_UNUSED real rc31, rc32, rc33, rc51, rc52, rc53, rc71, rc72, rc73;
   if CONSTEXPR (!do_g) {
      damp_aplus3(r, pdi, ddi, pdk, ddk, //
                  ex3, ex5, ex7);
      ex3 = 1 - ex3;
      ex5 = 1 - ex5;
      ex7 = 1 - ex7;
   } else {
      damp_aplus3g(          //
         r, rr2, xr, yr, zr, //
         pdi, ddi, pdk, ddk, //
         ex3, ex5, ex7,      //
         rc31, rc32, rc33,   //
         rc51, rc52, rc53,   //
         rc71, rc72, rc73);
      rc31 *= rr3;
      rc32 *= rr3;
      rc33 *= rr3;
      rc51 *= rr5;
      rc52 *= rr5;
      rc53 *= rr5;
      rc71 *= rr7;
      rc72 *= rr7;
      rc73 *= rr7;
   }
   // end if use_dirdamp

   real sr3 = bn[1] - ex3 * rr3;
   real sr5 = bn[2] - ex5 * rr5;
   real sr7 = bn[3] - ex7 * rr7;

   if CONSTEXPR (do_e) {
      real diu = dix * ukx + diy * uky + diz * ukz;
      real qiu = qix * ukx + qiy * uky + qiz * ukz;
      real dku = dkx * uix + dky * uiy + dkz * uiz;
      real qku = qkx * uix + qky * uiy + qkz * uiz;
      real term1 = ck * uir - ci * ukr + diu + dku;
      real term2 = 2 * (qiu - qku) - uir * dkr - dir * ukr;
      real term3 = uir * qkr - ukr * qir;
      e = dscale * f * (term1 * sr3 + term2 * sr5 + term3 * sr7);
   }

   if CONSTEXPR (CFLX) {
      poti = -2.0 * ukr * f * dscale * sr3;
      potk = 2.0 * uir * f * dscale * sr3;
   }

   if CONSTEXPR (do_g) {
      real uir = uix * xr + uiy * yr + uiz * zr;
      real ukr = ukx * xr + uky * yr + ukz * zr;

      // get the induced dipole field used for dipole torques

      real tuir, tukr;

      real tix3 = 2.0 * dscale * sr3 * ukx;
      real tiy3 = 2.0 * dscale * sr3 * uky;
      real tiz3 = 2.0 * dscale * sr3 * ukz;
      real tkx3 = 2.0 * dscale * sr3 * uix;
      real tky3 = 2.0 * dscale * sr3 * uiy;
      real tkz3 = 2.0 * dscale * sr3 * uiz;
      tuir = -2.0 * dscale * sr5 * ukr;
      tukr = -2.0 * dscale * sr5 * uir;

      uf0i += f * (tix3 + xr * tuir);
      uf1i += f * (tiy3 + yr * tuir);
      uf2i += f * (tiz3 + zr * tuir);
      uf0k += f * (tkx3 + xr * tukr);
      uf1k += f * (tky3 + yr * tukr);
      uf2k += f * (tkz3 + zr * tukr);

      // get induced dipole field gradient used for quadrupole torques

      real tix5 = 4 * (dscale * sr5 * ukx);
      real tiy5 = 4 * (dscale * sr5 * uky);
      real tiz5 = 4 * (dscale * sr5 * ukz);
      real tkx5 = 4 * (dscale * sr5 * uix);
      real tky5 = 4 * (dscale * sr5 * uiy);
      real tkz5 = 4 * (dscale * sr5 * uiz);
      tuir = -2.0 * dscale * sr7 * ukr;
      tukr = -2.0 * dscale * sr7 * uir;

      duf0i += f * (xr * tix5 + xr * xr * tuir);
      duf1i += f * (xr * tiy5 + yr * tix5 + 2 * xr * yr * tuir);
      duf2i += f * (yr * tiy5 + yr * yr * tuir);
      duf3i += f * (xr * tiz5 + zr * tix5 + 2 * xr * zr * tuir);
      duf4i += f * (yr * tiz5 + zr * tiy5 + 2 * yr * zr * tuir);
      duf5i += f * (zr * tiz5 + zr * zr * tuir);
      duf0k += f * (-xr * tkx5 - xr * xr * tukr);
      duf1k += f * (-xr * tky5 - yr * tkx5 - 2 * xr * yr * tukr);
      duf2k += f * (-yr * tky5 - yr * yr * tukr);
      duf3k += f * (-xr * tkz5 - zr * tkx5 - 2 * xr * zr * tukr);
      duf4k += f * (-yr * tkz5 - zr * tky5 - 2 * yr * zr * tukr);
      duf5k += f * (-zr * tkz5 - zr * zr * tukr);

      // get the field gradient for direct polarization force

      real term1, term2, term3, term4, term5, term6, term7;

      term1 = bn[2] - ex3 * rr5;
      term2 = bn[3] - ex5 * rr7;
      term3 = -sr3 + term1 * xr * xr - xr * rc31;
      term4 = rc31 - term1 * xr - sr5 * xr;
      term5 = term2 * xr * xr - sr5 - xr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * xr - bn[3] - xr * rc71;
      term7 = rc51 - 2 * bn[3] * xr + (ex5 + 1.5f * ex7) * rr7 * xr;
      real tixx = ci * term3 + dix * term4 + dir * term5 + 2 * sr5 * qixx + (qiy * yr + qiz * zr) * ex7 * rr7 + 2 * qix * term7 + qir * term6;
      real tkxx = ck * term3 - dkx * term4 - dkr * term5 + 2 * sr5 * qkxx + (qky * yr + qkz * zr) * ex7 * rr7 + 2 * qkx * term7 + qkr * term6;

      term3 = -sr3 + term1 * yr * yr - yr * rc32;
      term4 = rc32 - term1 * yr - sr5 * yr;
      term5 = term2 * yr * yr - sr5 - yr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * yr - bn[3] - yr * rc72;
      term7 = rc52 - 2 * bn[3] * yr + (ex5 + 1.5f * ex7) * rr7 * yr;
      real tiyy = ci * term3 + diy * term4 + dir * term5 + 2 * sr5 * qiyy + (qix * xr + qiz * zr) * ex7 * rr7 + 2 * qiy * term7 + qir * term6;
      real tkyy = ck * term3 - dky * term4 - dkr * term5 + 2 * sr5 * qkyy + (qkx * xr + qkz * zr) * ex7 * rr7 + 2 * qky * term7 + qkr * term6;

      term3 = -sr3 + term1 * zr * zr - zr * rc33;
      term4 = rc33 - term1 * zr - sr5 * zr;
      term5 = term2 * zr * zr - sr5 - zr * rc53;
      term6 = (bn[4] - ex7 * rr9) * zr * zr - bn[3] - zr * rc73;
      term7 = rc53 - 2 * bn[3] * zr + (ex5 + 1.5f * ex7) * rr7 * zr;
      real tizz = ci * term3 + diz * term4 + dir * term5 + 2 * sr5 * qizz + (qix * xr + qiy * yr) * ex7 * rr7 + 2 * qiz * term7 + qir * term6;
      real tkzz = ck * term3 - dkz * term4 - dkr * term5 + 2 * sr5 * qkzz + (qkx * xr + qky * yr) * ex7 * rr7 + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * xr * yr - yr * rc31;
      term4 = rc31 - term1 * xr;
      term5 = term2 * xr * yr - yr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * yr - yr * rc71;
      term7 = rc51 - term2 * xr;
      real tixy = ci * term3 - sr5 * dix * yr + diy * term4 + dir * term5 + 2 * sr5 * qixy - 2 * sr7 * yr * qix + 2 * qiy * term7 + qir * term6;
      real tkxy = ck * term3 + sr5 * dkx * yr - dky * term4 - dkr * term5 + 2 * sr5 * qkxy - 2 * sr7 * yr * qkx + 2 * qky * term7 + qkr * term6;

      term3 = term1 * xr * zr - zr * rc31;
      term5 = term2 * xr * zr - zr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * zr - zr * rc71;
      real tixz = ci * term3 - sr5 * dix * zr + diz * term4 + dir * term5 + 2 * sr5 * qixz - 2 * sr7 * zr * qix + 2 * qiz * term7 + qir * term6;
      real tkxz = ck * term3 + sr5 * dkx * zr - dkz * term4 - dkr * term5 + 2 * sr5 * qkxz - 2 * sr7 * zr * qkx + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * yr * zr - zr * rc32;
      term4 = rc32 - term1 * yr;
      term5 = term2 * yr * zr - zr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * zr - zr * rc72;
      term7 = rc52 - term2 * yr;
      real tiyz = ci * term3 - sr5 * diy * zr + diz * term4 + dir * term5 + 2 * sr5 * qiyz - 2 * sr7 * zr * qiy + 2 * qiz * term7 + qir * term6;
      real tkyz = ck * term3 + sr5 * dky * zr - dkz * term4 - dkr * term5 + 2 * sr5 * qkyz - 2 * sr7 * zr * qky + 2 * qkz * term7 + qkr * term6;

      // get the dEd/dR terms for Thole direct polarization force

      real depx, depy, depz;
      real frcx, frcy, frcz;

      depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy - tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy - tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy - tkzz * uiz;
      if CONSTEXPR (eq<ETYP, EWALD>()) {
         frcx = -2 * depx;
         frcy = -2 * depy;
         frcz = -2 * depz;
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         frcx = -2 * depx * dscale;
         frcy = -2 * depy * dscale;
         frcz = -2 * depz * dscale;
      }

      // get the dtau/dr terms used for mutual polarization force
      damp_thole3g(          //
         r, rr2, xr, yr, zr, //
         pdi, pti, pdk, ptk, //
         ex3, ex5, ex7,      //
         rc31, rc32, rc33,   //
         rc51, rc52, rc53,   //
         rc71, rc72, rc73);
      rc31 *= rr3;
      rc32 *= rr3;
      rc33 *= rr3;
      rc51 *= rr5;
      rc52 *= rr5;
      rc53 *= rr5;
      sr3 = bn[1] - ex3 * rr3;
      sr5 = bn[2] - ex5 * rr5;

      term1 = bn[2] - ex3 * rr5;
      term2 = bn[3] - ex5 * rr7;
      term3 = sr5 + term1;

      term5 = -xr * term3 + rc31;
      term6 = -sr5 + xr * xr * term2 - xr * rc51;
      tixx = uix * term5 + uir * term6;
      tkxx = ukx * term5 + ukr * term6;

      term5 = -yr * term3 + rc32;
      term6 = -sr5 + yr * yr * term2 - yr * rc52;
      tiyy = uiy * term5 + uir * term6;
      tkyy = uky * term5 + ukr * term6;

      term5 = -zr * term3 + rc33;
      term6 = -sr5 + zr * zr * term2 - zr * rc53;
      tizz = uiz * term5 + uir * term6;
      tkzz = ukz * term5 + ukr * term6;

      term4 = -sr5 * yr;
      term5 = -xr * term1 + rc31;
      term6 = xr * yr * term2 - yr * rc51;
      tixy = uix * term4 + uiy * term5 + uir * term6;
      tkxy = ukx * term4 + uky * term5 + ukr * term6;

      term4 = -sr5 * zr;
      term6 = xr * zr * term2 - zr * rc51;
      tixz = uix * term4 + uiz * term5 + uir * term6;
      tkxz = ukx * term4 + ukz * term5 + ukr * term6;

      term5 = -yr * term1 + rc32;
      term6 = yr * zr * term2 - zr * rc52;
      tiyz = uiy * term4 + uiz * term5 + uir * term6;
      tkyz = uky * term4 + ukz * term5 + ukr * term6;

      depx = tixx * ukx + tixy * uky + tixz * ukz + tkxx * uix + tkxy * uiy + tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz + tkxy * uix + tkyy * uiy + tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz + tkxz * uix + tkyz * uiy + tkzz * uiz;
      if CONSTEXPR (eq<ETYP, EWALD>()) {
         frcx -= depx;
         frcy -= depy;
         frcz -= depz;
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         frcx -= uscale * depx;
         frcy -= uscale * depy;
         frcz -= uscale * depz;
      }

      frcx *= f;
      frcy *= f;
      frcz *= f;
      frcxi += frcx;
      frcyi += frcy;
      frczi += frcz;
      frcxk -= frcx;
      frcyk -= frcy;
      frczk -= frcz;

      if CONSTEXPR (do_v) {
         vxx = -xr * frcx;
         vxy = -0.5f * (yr * frcx + xr * frcy);
         vxz = -0.5f * (zr * frcx + xr * frcz);
         vyy = -yr * frcy;
         vyz = -0.5f * (zr * frcy + yr * frcz);
         vzz = -zr * frcz;
      }
   }
}
}
