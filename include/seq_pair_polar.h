#pragma once
#include "elec.h"
#include "md.h"
#include "seq_damp.h"


TINKER_NAMESPACE_BEGIN
struct PairPolarGrad
{
   real frcx, frcy, frcz;
   real ufldi[3], ufldk[3];
   real dufldi[6], dufldk[6];
};


SEQ_ROUTINE
inline void zero(PairPolarGrad& pgrad)
{
   pgrad.frcx = 0;
   pgrad.frcy = 0;
   pgrad.frcz = 0;
   #pragma unroll
   for (int i = 0; i < 3; ++i) {
      pgrad.ufldi[i] = 0;
   }
   #pragma unroll
   for (int i = 0; i < 3; ++i) {
      pgrad.ufldk[i] = 0;
   }
   #pragma unroll
   for (int i = 0; i < 6; ++i) {
      pgrad.dufldi[i] = 0;
   }
   #pragma unroll
   for (int i = 0; i < 6; ++i) {
      pgrad.dufldk[i] = 0;
   }
}


#pragma acc routine seq
template <int USE, elec_t ETYP>
SEQ_CUDA
void pair_polar(                                                              //
   real r2, real xr, real yr, real zr, real dscale, real pscale, real uscale, //
   real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
   real qiyy, real qiyz, real qizz, real uix, real uiy, real uiz, real uixp,
   real uiyp, real uizp, real pdi, real pti, //
   real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
   real qkyy, real qkyz, real qkzz, real ukx, real uky, real ukz, real ukxp,
   real ukyp, real ukzp, real pdk, real ptk, //
   real f, real aewald, real& restrict e, PairPolarGrad& restrict pgrad)
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_a = USE & calc::analyz;

   if CONSTEXPR (ETYP == elec_t::ewald) {
      dscale = 1;
      pscale = 1;
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
   if CONSTEXPR (ETYP == elec_t::ewald) {
      if CONSTEXPR (!do_g)
         damp_ewald<4>(bn, r, invr1, rr2, aewald);
      else
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
   } else if CONSTEXPR (ETYP == elec_t::coulomb) {
      bn[1] = rr3;
      bn[2] = rr5;
      bn[3] = rr7;
      if CONSTEXPR (do_g)
         bn[4] = rr9;
   }

   // if use_thole
   real ex3, ex5, ex7;
   MAYBE_UNUSED real rc31, rc32, rc33, rc51, rc52, rc53, rc71, rc72, rc73;
   if CONSTEXPR (!do_g) {
      damp_thole3(r, pdi, pti, pdk, ptk, //
                  ex3, ex5, ex7);
      ex3 = 1 - ex3;
      ex5 = 1 - ex5;
      ex7 = 1 - ex7;
   } else {
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
      rc71 *= rr7;
      rc72 *= rr7;
      rc73 *= rr7;
   }
   // end if use_thole

   real sr3 = bn[1] - ex3 * rr3;
   real sr5 = bn[2] - ex5 * rr5;
   real sr7 = bn[3] - ex7 * rr7;

   if CONSTEXPR (do_e && do_a) {
      real diu = dix * ukx + diy * uky + diz * ukz;
      real qiu = qix * ukx + qiy * uky + qiz * ukz;
      real dku = dkx * uix + dky * uiy + dkz * uiz;
      real qku = qkx * uix + qky * uiy + qkz * uiz;
      real term1 = ck * uir - ci * ukr + diu + dku;
      real term2 = 2 * (qiu - qku) - uir * dkr - dir * ukr;
      real term3 = uir * qkr - ukr * qir;
      e = pscale * f * (term1 * sr3 + term2 * sr5 + term3 * sr7);
   }

   if CONSTEXPR (do_g) {
      real uirp = uixp * xr + uiyp * yr + uizp * zr;
      real ukrp = ukxp * xr + ukyp * yr + ukzp * zr;

      // get the induced dipole field used for dipole torques

      real tuir, tukr;

      real tix3 = pscale * sr3 * ukx + dscale * sr3 * ukxp;
      real tiy3 = pscale * sr3 * uky + dscale * sr3 * ukyp;
      real tiz3 = pscale * sr3 * ukz + dscale * sr3 * ukzp;
      real tkx3 = pscale * sr3 * uix + dscale * sr3 * uixp;
      real tky3 = pscale * sr3 * uiy + dscale * sr3 * uiyp;
      real tkz3 = pscale * sr3 * uiz + dscale * sr3 * uizp;
      tuir = -pscale * sr5 * ukr - dscale * sr5 * ukrp;
      tukr = -pscale * sr5 * uir - dscale * sr5 * uirp;

      pgrad.ufldi[0] = f * (tix3 + xr * tuir);
      pgrad.ufldi[1] = f * (tiy3 + yr * tuir);
      pgrad.ufldi[2] = f * (tiz3 + zr * tuir);
      pgrad.ufldk[0] = f * (tkx3 + xr * tukr);
      pgrad.ufldk[1] = f * (tky3 + yr * tukr);
      pgrad.ufldk[2] = f * (tkz3 + zr * tukr);

      // get induced dipole field gradient used for quadrupole torques

      real tix5 = 2 * (pscale * sr5 * ukx + dscale * sr5 * ukxp);
      real tiy5 = 2 * (pscale * sr5 * uky + dscale * sr5 * ukyp);
      real tiz5 = 2 * (pscale * sr5 * ukz + dscale * sr5 * ukzp);
      real tkx5 = 2 * (pscale * sr5 * uix + dscale * sr5 * uixp);
      real tky5 = 2 * (pscale * sr5 * uiy + dscale * sr5 * uiyp);
      real tkz5 = 2 * (pscale * sr5 * uiz + dscale * sr5 * uizp);
      tuir = -pscale * sr7 * ukr - dscale * sr7 * ukrp;
      tukr = -pscale * sr7 * uir - dscale * sr7 * uirp;

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
      real tixx = ci * term3 + dix * term4 + dir * term5 + 2 * sr5 * qixx +
         (qiy * yr + qiz * zr) * ex7 * rr7 + 2 * qix * term7 + qir * term6;
      real tkxx = ck * term3 - dkx * term4 - dkr * term5 + 2 * sr5 * qkxx +
         (qky * yr + qkz * zr) * ex7 * rr7 + 2 * qkx * term7 + qkr * term6;

      term3 = -sr3 + term1 * yr * yr - yr * rc32;
      term4 = rc32 - term1 * yr - sr5 * yr;
      term5 = term2 * yr * yr - sr5 - yr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * yr - bn[3] - yr * rc72;
      term7 = rc52 - 2 * bn[3] * yr + (ex5 + 1.5f * ex7) * rr7 * yr;
      real tiyy = ci * term3 + diy * term4 + dir * term5 + 2 * sr5 * qiyy +
         (qix * xr + qiz * zr) * ex7 * rr7 + 2 * qiy * term7 + qir * term6;
      real tkyy = ck * term3 - dky * term4 - dkr * term5 + 2 * sr5 * qkyy +
         (qkx * xr + qkz * zr) * ex7 * rr7 + 2 * qky * term7 + qkr * term6;

      term3 = -sr3 + term1 * zr * zr - zr * rc33;
      term4 = rc33 - term1 * zr - sr5 * zr;
      term5 = term2 * zr * zr - sr5 - zr * rc53;
      term6 = (bn[4] - ex7 * rr9) * zr * zr - bn[3] - zr * rc73;
      term7 = rc53 - 2 * bn[3] * zr + (ex5 + 1.5f * ex7) * rr7 * zr;
      real tizz = ci * term3 + diz * term4 + dir * term5 + 2 * sr5 * qizz +
         (qix * xr + qiy * yr) * ex7 * rr7 + 2 * qiz * term7 + qir * term6;
      real tkzz = ck * term3 - dkz * term4 - dkr * term5 + 2 * sr5 * qkzz +
         (qkx * xr + qky * yr) * ex7 * rr7 + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * xr * yr - yr * rc31;
      term4 = rc31 - term1 * xr;
      term5 = term2 * xr * yr - yr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * yr - yr * rc71;
      term7 = rc51 - term2 * xr;
      real tixy = ci * term3 - sr5 * dix * yr + diy * term4 + dir * term5 +
         2 * sr5 * qixy - 2 * sr7 * yr * qix + 2 * qiy * term7 + qir * term6;
      real tkxy = ck * term3 + sr5 * dkx * yr - dky * term4 - dkr * term5 +
         2 * sr5 * qkxy - 2 * sr7 * yr * qkx + 2 * qky * term7 + qkr * term6;

      term3 = term1 * xr * zr - zr * rc31;
      term5 = term2 * xr * zr - zr * rc51;
      term6 = (bn[4] - ex7 * rr9) * xr * zr - zr * rc71;
      real tixz = ci * term3 - sr5 * dix * zr + diz * term4 + dir * term5 +
         2 * sr5 * qixz - 2 * sr7 * zr * qix + 2 * qiz * term7 + qir * term6;
      real tkxz = ck * term3 + sr5 * dkx * zr - dkz * term4 - dkr * term5 +
         2 * sr5 * qkxz - 2 * sr7 * zr * qkx + 2 * qkz * term7 + qkr * term6;

      term3 = term1 * yr * zr - zr * rc32;
      term4 = rc32 - term1 * yr;
      term5 = term2 * yr * zr - zr * rc52;
      term6 = (bn[4] - ex7 * rr9) * yr * zr - zr * rc72;
      term7 = rc52 - term2 * yr;
      real tiyz = ci * term3 - sr5 * diy * zr + diz * term4 + dir * term5 +
         2 * sr5 * qiyz - 2 * sr7 * zr * qiy + 2 * qiz * term7 + qir * term6;
      real tkyz = ck * term3 + sr5 * dky * zr - dkz * term4 - dkr * term5 +
         2 * sr5 * qkyz - 2 * sr7 * zr * qky + 2 * qkz * term7 + qkr * term6;

      // get the dEd/dR terms for Thole direct polarization force

      real depx, depy, depz;

      depx = tixx * ukxp + tixy * ukyp + tixz * ukzp - tkxx * uixp -
         tkxy * uiyp - tkxz * uizp;
      depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp - tkxy * uixp -
         tkyy * uiyp - tkyz * uizp;
      depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp - tkxz * uixp -
         tkyz * uiyp - tkzz * uizp;
      if CONSTEXPR (ETYP == elec_t::ewald) {
         pgrad.frcx = -depx;
         pgrad.frcy = -depy;
         pgrad.frcz = -depz;
      } else if CONSTEXPR (ETYP == elec_t::coulomb) {
         pgrad.frcx = -depx * dscale;
         pgrad.frcy = -depy * dscale;
         pgrad.frcz = -depz * dscale;
      }

      // get the dEp/dR terms for Thole direct polarization force

      depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy -
         tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy -
         tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy -
         tkzz * uiz;
      if CONSTEXPR (ETYP == elec_t::ewald) {
         pgrad.frcx -= depx;
         pgrad.frcy -= depy;
         pgrad.frcz -= depz;
      } else if CONSTEXPR (ETYP == elec_t::coulomb) {
         pgrad.frcx -= pscale * depx;
         pgrad.frcy -= pscale * depy;
         pgrad.frcz -= pscale * depz;
      }

      // get the dtau/dr terms used for mutual polarization force

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

      depx = tixx * ukxp + tixy * ukyp + tixz * ukzp + tkxx * uixp +
         tkxy * uiyp + tkxz * uizp;
      depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp + tkxy * uixp +
         tkyy * uiyp + tkyz * uizp;
      depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp + tkxz * uixp +
         tkyz * uiyp + tkzz * uizp;
      if CONSTEXPR (ETYP == elec_t::ewald) {
         pgrad.frcx -= depx;
         pgrad.frcy -= depy;
         pgrad.frcz -= depz;
      } else if CONSTEXPR (ETYP == elec_t::coulomb) {
         pgrad.frcx -= uscale * depx;
         pgrad.frcy -= uscale * depy;
         pgrad.frcz -= uscale * depz;
      }

      pgrad.frcx *= f;
      pgrad.frcy *= f;
      pgrad.frcz *= f;
   }
}
TINKER_NAMESPACE_END
