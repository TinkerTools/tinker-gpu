#pragma once
#include "seq/damp_hippo.h"
#include "seq/pair_polar.h"

namespace tinker {
#pragma acc routine seq
template <bool do_e, bool do_g, class ETYP, int CFLX>
SEQ_CUDA
void pair_polar_chgpen(real r2,
                       real xr,
                       real yr,
                       real zr,
                       real dscale,
                       real wscale,
                       real ci,
                       real dix,
                       real diy,
                       real diz,
                       real corei,
                       real vali,
                       real alphai,
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
                       real corek,
                       real valk,
                       real alphak,
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

   real rr1 = f * invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   real rr9 = 7 * rr7 * rr2;

   real bn[5];
   real dmpi[5], dmpk[5];
   real dmpik[6];

   real rr3core, rr5core, rr3i, rr5i, rr7i, rr9i;
   real rr3k, rr5k, rr7k, rr9k, rr5ik, rr7ik;
   real dsr3i, dsr5i, dsr7i, dsr3k, dsr5k, dsr7k;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (do_g) {
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
         damp_pole_v2<11>(dmpik, dmpi, dmpk, r, alphai, alphak);
      } else {
         damp_ewald<4>(bn, r, invr1, rr2, aewald);
         damp_pole_v2<9>(dmpik, dmpi, dmpk, r, alphai, alphak);
      }

      bn[1] *= f;
      bn[2] *= f;
      bn[3] *= f;
      bn[4] *= f;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      if CONSTEXPR (do_g)
         damp_pole_v2<11>(dmpik, dmpi, dmpk, r, alphai, alphak);
      else {
         damp_pole_v2<9>(dmpik, dmpi, dmpk, r, alphai, alphak);
      }

      bn[1] = rr3;
      bn[2] = rr5;
      bn[3] = rr7;
      bn[4] = rr9;
   }

   rr3core = bn[1] - (1 - dscale) * rr3;
   rr5core = bn[2] - (1 - dscale) * rr5;
   rr3i = bn[1] - (1 - dscale * dmpi[1]) * rr3;
   rr5i = bn[2] - (1 - dscale * dmpi[2]) * rr5;
   rr7i = bn[3] - (1 - dscale * dmpi[3]) * rr7;
   rr9i = bn[4] - (1 - dscale * dmpi[4]) * rr9;
   rr3k = bn[1] - (1 - dscale * dmpk[1]) * rr3;
   rr5k = bn[2] - (1 - dscale * dmpk[2]) * rr5;
   rr7k = bn[3] - (1 - dscale * dmpk[3]) * rr7;
   rr9k = bn[4] - (1 - dscale * dmpk[4]) * rr9;
   rr5ik = bn[2] - (1 - wscale * dmpik[2]) * rr5;
   rr7ik = bn[3] - (1 - wscale * dmpik[3]) * rr7;

   dsr3i = 2 * rr3i;
   dsr3k = 2 * rr3k;
   dsr5i = 2 * rr5i;
   dsr5k = 2 * rr5k;
   dsr7i = 2 * rr7i;
   dsr7k = 2 * rr7k;

   if CONSTEXPR (do_e) {
      real diu = dix * ukx + diy * uky + diz * ukz;
      real qiu = qix * ukx + qiy * uky + qiz * ukz;
      real dku = dkx * uix + dky * uiy + dkz * uiz;
      real qku = qkx * uix + qky * uiy + qkz * uiz;
      e = uir * (corek * rr3core + valk * rr3k) - ukr * (corei * rr3core + vali * rr3i) + diu * rr3i + dku * rr3k + 2 * (qiu * rr5i - qku * rr5k)
         - dkr * uir * rr5k - dir * ukr * rr5i + qkr * uir * rr7k - qir * ukr * rr7i;
   }

   if CONSTEXPR (do_g) {
      // get the induced dipole field used for dipole torques
      real tix3 = dsr3i * ukx;
      real tiy3 = dsr3i * uky;
      real tiz3 = dsr3i * ukz;
      real tkx3 = dsr3k * uix;
      real tky3 = dsr3k * uiy;
      real tkz3 = dsr3k * uiz;
      real tuir = -dsr5i * ukr;
      real tukr = -dsr5k * uir;

      if CONSTEXPR (CFLX) {
         poti = -ukr * dsr3i;
         potk = uir * dsr3k;
      }

      pgrad.ufldi[0] = (tix3 + xr * tuir);
      pgrad.ufldi[1] = (tiy3 + yr * tuir);
      pgrad.ufldi[2] = (tiz3 + zr * tuir);
      pgrad.ufldk[0] = (tkx3 + xr * tukr);
      pgrad.ufldk[1] = (tky3 + yr * tukr);
      pgrad.ufldk[2] = (tkz3 + zr * tukr);

      // get induced dipole field gradient used for quadrupole torques

      real tix5 = 2 * (dsr5i * ukx);
      real tiy5 = 2 * (dsr5i * uky);
      real tiz5 = 2 * (dsr5i * ukz);
      real tkx5 = 2 * (dsr5k * uix);
      real tky5 = 2 * (dsr5k * uiy);
      real tkz5 = 2 * (dsr5k * uiz);
      tuir = -dsr7i * ukr;
      tukr = -dsr7k * uir;

      pgrad.dufldi[0] = (xr * tix5 + xr * xr * tuir);
      pgrad.dufldi[1] = (xr * tiy5 + yr * tix5 + 2 * xr * yr * tuir);
      pgrad.dufldi[2] = (yr * tiy5 + yr * yr * tuir);
      pgrad.dufldi[3] = (xr * tiz5 + zr * tix5 + 2 * xr * zr * tuir);
      pgrad.dufldi[4] = (yr * tiz5 + zr * tiy5 + 2 * yr * zr * tuir);
      pgrad.dufldi[5] = (zr * tiz5 + zr * zr * tuir);
      pgrad.dufldk[0] = (-xr * tkx5 - xr * xr * tukr);
      pgrad.dufldk[1] = (-xr * tky5 - yr * tkx5 - 2 * xr * yr * tukr);
      pgrad.dufldk[2] = (-yr * tky5 - yr * yr * tukr);
      pgrad.dufldk[3] = (-xr * tkz5 - zr * tkx5 - 2 * xr * zr * tukr);
      pgrad.dufldk[4] = (-yr * tkz5 - zr * tky5 - 2 * yr * zr * tukr);
      pgrad.dufldk[5] = (-zr * tkz5 - zr * zr * tukr);

      // get the field gradient for direct polarization force

      real term1i, term2i, term3i, term4i, term5i, term6i, term1core;
      real term7i, term8i;
      real term1k, term2k, term3k, term4k, term5k, term6k;
      real term7k, term8k;

      term1i = rr3i - rr5i * xr * xr;
      term1core = rr3core - rr5core * xr * xr;
      term2i = 2 * rr5i * xr;
      term3i = rr7i * xr * xr - rr5i;
      term4i = 2 * rr5i;
      term5i = 5 * rr7i * xr;
      term6i = rr9i * xr * xr;
      term1k = rr3k - rr5k * xr * xr;
      term2k = 2 * rr5k * xr;
      term3k = rr7k * xr * xr - rr5k;
      term4k = 2 * rr5k;
      term5k = 5 * rr7k * xr;
      term6k = rr9k * xr * xr;
      real tixx = vali * term1i + corei * term1core + dix * term2i - dir * term3i - qixx * term4i + qix * term5i - qir * term6i + (qiy * yr + qiz * zr) * rr7i;
      real tkxx = valk * term1k + corek * term1core - dkx * term2k + dkr * term3k - qkxx * term4k + qkx * term5k - qkr * term6k + (qky * yr + qkz * zr) * rr7k;
      term1i = rr3i - rr5i * yr * yr;
      term1core = rr3core - rr5core * yr * yr;
      term2i = 2 * rr5i * yr;
      term3i = rr7i * yr * yr - rr5i;
      term4i = 2 * rr5i;
      term5i = 5 * rr7i * yr;
      term6i = rr9i * yr * yr;
      term1k = rr3k - rr5k * yr * yr;
      term2k = 2 * rr5k * yr;
      term3k = rr7k * yr * yr - rr5k;
      term4k = 2 * rr5k;
      term5k = 5 * rr7k * yr;
      term6k = rr9k * yr * yr;
      real tiyy = vali * term1i + corei * term1core + diy * term2i - dir * term3i - qiyy * term4i + qiy * term5i - qir * term6i + (qix * xr + qiz * zr) * rr7i;
      real tkyy = valk * term1k + corek * term1core - dky * term2k + dkr * term3k - qkyy * term4k + qky * term5k - qkr * term6k + (qkx * xr + qkz * zr) * rr7k;
      term1i = rr3i - rr5i * zr * zr;
      term1core = rr3core - rr5core * zr * zr;
      term2i = 2 * rr5i * zr;
      term3i = rr7i * zr * zr - rr5i;
      term4i = 2 * rr5i;
      term5i = 5 * rr7i * zr;
      term6i = rr9i * zr * zr;
      term1k = rr3k - rr5k * zr * zr;
      term2k = 2 * rr5k * zr;
      term3k = rr7k * zr * zr - rr5k;
      term4k = 2 * rr5k;
      term5k = 5 * rr7k * zr;
      term6k = rr9k * zr * zr;
      real tizz = vali * term1i + corei * term1core + diz * term2i - dir * term3i - qizz * term4i + qiz * term5i - qir * term6i + (qix * xr + qiy * yr) * rr7i;
      real tkzz = valk * term1k + corek * term1core - dkz * term2k + dkr * term3k - qkzz * term4k + qkz * term5k - qkr * term6k + (qkx * xr + qky * yr) * rr7k;
      term2i = rr5i * xr;
      term1i = yr * term2i;
      term1core = rr5core * xr * yr;
      term3i = rr5i * yr;
      term4i = yr * (rr7i * xr);
      term5i = 2 * rr5i;
      term6i = 2 * rr7i * xr;
      term7i = 2 * rr7i * yr;
      term8i = yr * rr9i * xr;
      term2k = rr5k * xr;
      term1k = yr * term2k;
      term3k = rr5k * yr;
      term4k = yr * (rr7k * xr);
      term5k = 2 * rr5k;
      term6k = 2 * rr7k * xr;
      term7k = 2 * rr7k * yr;
      term8k = yr * rr9k * xr;
      real tixy = -vali * term1i - corei * term1core + diy * term2i + dix * term3i - dir * term4i - qixy * term5i + qiy * term6i + qix * term7i - qir * term8i;
      real tkxy = -valk * term1k - corek * term1core - dky * term2k - dkx * term3k + dkr * term4k - qkxy * term5k + qky * term6k + qkx * term7k - qkr * term8k;

      term2i = rr5i * xr;
      term1i = zr * term2i;
      term1core = rr5core * xr * zr;
      term3i = rr5i * zr;
      term4i = zr * (rr7i * xr);
      term5i = 2 * rr5i;
      term6i = 2 * rr7i * xr;
      term7i = 2 * rr7i * zr;
      term8i = zr * rr9i * xr;
      term2k = rr5k * xr;
      term1k = zr * term2k;
      term3k = rr5k * zr;
      term4k = zr * (rr7k * xr);
      term5k = 2 * rr5k;
      term6k = 2 * rr7k * xr;
      term7k = 2 * rr7k * zr;
      term8k = zr * rr9k * xr;
      real tixz = -vali * term1i - corei * term1core + diz * term2i + dix * term3i - dir * term4i - qixz * term5i + qiz * term6i + qix * term7i - qir * term8i;
      real tkxz = -valk * term1k - corek * term1core - dkz * term2k - dkx * term3k + dkr * term4k - qkxz * term5k + qkz * term6k + qkx * term7k - qkr * term8k;

      term2i = rr5i * yr;
      term1i = zr * term2i;
      term1core = rr5core * yr * zr;
      term3i = rr5i * zr;
      term4i = zr * (rr7i * yr);
      term5i = 2 * rr5i;
      term6i = 2 * rr7i * yr;
      term7i = 2 * rr7i * zr;
      term8i = zr * rr9i * yr;
      term2k = rr5k * yr;
      term1k = zr * term2k;
      term3k = rr5k * zr;
      term4k = zr * (rr7k * yr);
      term5k = 2 * rr5k;
      term6k = 2 * rr7k * yr;
      term7k = 2 * rr7k * zr;
      term8k = zr * rr9k * yr;
      real tiyz = -vali * term1i - corei * term1core + diz * term2i + diy * term3i - dir * term4i - qiyz * term5i + qiz * term6i + qiy * term7i - qir * term8i;
      real tkyz = -valk * term1k - corek * term1core - dkz * term2k - dky * term3k + dkr * term4k - qkyz * term5k + qkz * term6k + qky * term7k - qkr * term8k;

      real depx, depy, depz;

      depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy - tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy - tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy - tkzz * uiz;

      pgrad.frcx = 2 * depx;
      pgrad.frcy = 2 * depy;
      pgrad.frcz = 2 * depz;

      // get the dtau/dr terms used for mutual polarization force

      real term1 = 2 * rr5ik;
      real term2 = term1 * xr;
      real term3 = rr5ik - rr7ik * xr * xr;
      tixx = uix * term2 + uir * term3;
      tkxx = ukx * term2 + ukr * term3;
      term2 = term1 * yr;
      term3 = rr5ik - rr7ik * yr * yr;
      tiyy = uiy * term2 + uir * term3;
      tkyy = uky * term2 + ukr * term3;
      term2 = term1 * zr;
      term3 = rr5ik - rr7ik * zr * zr;
      tizz = uiz * term2 + uir * term3;
      tkzz = ukz * term2 + ukr * term3;
      term1 = rr5ik * yr;
      term2 = rr5ik * xr;
      term3 = yr * (rr7ik * xr);
      tixy = uix * term1 + uiy * term2 - uir * term3;
      tkxy = ukx * term1 + uky * term2 - ukr * term3;
      term1 = rr5ik * zr;
      term3 = zr * (rr7ik * xr);
      tixz = uix * term1 + uiz * term2 - uir * term3;
      tkxz = ukx * term1 + ukz * term2 - ukr * term3;
      term2 = rr5ik * yr;
      term3 = zr * (rr7ik * yr);
      tiyz = uiy * term1 + uiz * term2 - uir * term3;
      tkyz = uky * term1 + ukz * term2 - ukr * term3;

      depx = tixx * ukx + tixy * uky + tixz * ukz + tkxx * uix + tkxy * uiy + tkxz * uiz;
      depy = tixy * ukx + tiyy * uky + tiyz * ukz + tkxy * uix + tkyy * uiy + tkyz * uiz;
      depz = tixz * ukx + tiyz * uky + tizz * ukz + tkxz * uix + tkyz * uiy + tkzz * uiz;

      pgrad.frcx += depx;
      pgrad.frcy += depy;
      pgrad.frcz += depz;
   }
}
}
