#pragma once
#include "seq/dampaplus.h"
#include "seq/pair_mpole.h"

namespace tinker {
#pragma acc routine seq
template <bool do_e, bool do_g, class ETYP, int CFLX>
SEQ_CUDA
void pair_mpole_chgpen_aplus( //
   real r2,
   real xr,
   real yr,
   real zr,
   real mscale, //
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
   real qizz, //
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
   real qkzz, //
   real f,
   real aewald,
   real& restrict e,
   real& restrict poti,
   real& restrict potk,
   PairMPoleGrad& restrict pgrad)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real dmpi[5];
   real dmpk[5];
   real dmpik[6];
   real bn[6];

   real rr1 = f * invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   real rr9 = 7 * rr7 * rr2;
   real rr11;

   if CONSTEXPR (do_g)
      rr11 = 9 * rr9 * rr2;

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
   real dik = dix * dkx + diy * dky + diz * dkz;
   real qik = qix * qkx + qiy * qky + qiz * qkz;
   real diqk = dix * qkx + diy * qky + diz * qkz;
   real dkqi = dkx * qix + dky * qiy + dkz * qiz;
   real qiqk = 2 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz) + qixx * qkxx + qiyy * qkyy + qizz * qkzz;

   // chgpen_aplus terms
   real term4ik = dir * qkr - dkr * qir - 4 * qik;
   real term5ik = qir * qkr;
   real term6ik = 2 * (dkqi - diqk + qiqk) - dir * dkr;

   // Compute damping factors
   if CONSTEXPR (do_g) {
      damp_gordon2<11>(dmpik, dmpi, dmpk, r, alphai, alphak);
   } else {
      damp_gordon2<9>(dmpik, dmpi, dmpk, r, alphai, alphak);
   }
   //

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (do_g) {
         damp_ewald<6>(bn, r, invr1, rr2, aewald);
      } else {
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
      }

      bn[0] *= f;
      bn[1] *= f;
      bn[2] *= f;
      bn[3] *= f;
      bn[4] *= f;
      if CONSTEXPR (do_g)
         bn[5] *= f;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[0] = rr1;
      bn[1] = rr3;
      bn[2] = rr5;
      bn[3] = rr7;
      bn[4] = rr9;
      if CONSTEXPR (do_g)
         bn[5] = rr11;
   } // endif NON_EWALD

   // A damped factor is the scaled undamped factor rrNs less mscale*p*rrN, where
   // the penetration terms p (dmpi, dmpk, dmpik) decay like exp(-alpha*r). The
   // core-core, core-valence and valence-valence products are each thousands of
   // kcal/mol and cancel down to the total-charge interaction, so summed apart
   // they lose ~1e-3 kcal/mol in float. Grouped around the total charges, only
   // the small p terms are left to add.
   real m = 1 - mscale;
   real rr1s = bn[0] - m * rr1, w1 = mscale * rr1;
   real rr3s = bn[1] - m * rr3, w3 = mscale * rr3;
   real rr5s = bn[2] - m * rr5, w5 = mscale * rr5;
   real rr7s = bn[3] - m * rr7, w7 = mscale * rr7;
   real rr9s = bn[4] - m * rr9, w9 = mscale * rr9;
   real rr3ik = rr3s - dmpik[1] * w3;
   real rr5ik = rr5s - dmpik[2] * w5;
   real rr7ik = rr7s - dmpik[3] * w7;
   real rr9ik = rr9s - dmpik[4] * w9;

   // Total charges; corei + vali is what the separate products add up to.
   real qi = corei + vali;
   real qk = corek + valk;
   // Charge of k seen at i (fk), and of i seen at k (fi), through the damping
   // of the other site: fkN = corek*rrNi + valk*rrNik, fiN = corei*rrNk + vali*rrNik.
   real fk3 = qk * rr3s - w3 * (corek * dmpi[1] + valk * dmpik[1]);
   real fi3 = qi * rr3s - w3 * (corei * dmpk[1] + vali * dmpik[1]);
   real fk5 = qk * rr5s - w5 * (corek * dmpi[2] + valk * dmpik[2]);
   real fi5 = qi * rr5s - w5 * (corei * dmpk[2] + vali * dmpik[2]);

   if CONSTEXPR (do_e) {
      real cc1 = qi * qk * rr1s - w1 * (corek * vali * dmpi[0] + corei * valk * dmpk[0] + vali * valk * dmpik[0]);
      e = cc1 + dir * fk3 - dkr * fi3 + dik * rr3ik + qir * fk5 + qkr * fi5 + term6ik * rr5ik + term4ik * rr7ik + term5ik * rr9ik;
   }

   if CONSTEXPR (do_g) {
      // gradient
      real qixk = qixx * qkx + qixy * qky + qixz * qkz;
      real qiyk = qixy * qkx + qiyy * qky + qiyz * qkz;
      real qizk = qixz * qkx + qiyz * qky + qizz * qkz;
      real qkxi = qkxx * qix + qkxy * qiy + qkxz * qiz;
      real qkyi = qkxy * qix + qkyy * qiy + qkyz * qiz;
      real qkzi = qkxz * qix + qkyz * qiy + qkzz * qiz;

      real diqkx = dix * qkxx + diy * qkxy + diz * qkxz;
      real diqky = dix * qkxy + diy * qkyy + diz * qkyz;
      real diqkz = dix * qkxz + diy * qkyz + diz * qkzz;
      real dkqix = dkx * qixx + dky * qixy + dkz * qixz;
      real dkqiy = dkx * qixy + dky * qiyy + dkz * qiyz;
      real dkqiz = dkx * qixz + dky * qiyz + dkz * qizz;

      real rr11ik = bn[5] - m * rr11 - dmpik[5] * mscale * rr11;
      real fk7 = qk * rr7s - w7 * (corek * dmpi[3] + valk * dmpik[3]);
      real fi7 = qi * rr7s - w7 * (corei * dmpk[3] + vali * dmpik[3]);
      real cc3 = qi * qk * rr3s - w3 * (corek * vali * dmpi[1] + corei * valk * dmpk[1] + vali * valk * dmpik[1]);
      real de = cc3 + dir * fk5 - dkr * fi5 + dik * rr5ik + qir * fk7 + qkr * fi7 + term6ik * rr7ik + term4ik * rr9ik + term5ik * rr11ik;

      real term1 = -fk3 + dkr * rr5ik - qkr * rr7ik;
      real term2 = fi3 + dir * rr5ik + qir * rr7ik;
      real term3 = 2 * rr5ik;
      real term4 = -2 * (fk5 - dkr * rr7ik + qkr * rr9ik);
      real term5 = -2 * (fi5 + dir * rr7ik + qir * rr9ik);
      real term6 = 4 * rr7ik;

      if CONSTEXPR (CFLX) {
         real t1i = qk * rr1s - w1 * (corek * dmpi[0] + valk * dmpik[0]);
         real t1k = qi * rr1s - w1 * (corei * dmpk[0] + vali * dmpik[0]);
         real t2i = -dkr * rr3ik;
         real t2k = dir * rr3ik;
         real t3i = qkr * rr5ik;
         real t3k = qir * rr5ik;
         poti = t1i + t2i + t3i;
         potk = t1k + t2k + t3k;
      }

      pgrad.frcx = de * xr + term1 * dix + term2 * dkx + term3 * (diqkx - dkqix) + term4 * qix + term5 * qkx + term6 * (qixk + qkxi);

      pgrad.frcy = de * yr + term1 * diy + term2 * dky + term3 * (diqky - dkqiy) + term4 * qiy + term5 * qky + term6 * (qiyk + qkyi);
      pgrad.frcz = de * zr + term1 * diz + term2 * dkz + term3 * (diqkz - dkqiz) + term4 * qiz + term5 * qkz + term6 * (qizk + qkzi);

      // torque
      real dirx = diy * zr - diz * yr;
      real diry = diz * xr - dix * zr;
      real dirz = dix * yr - diy * xr;
      real dkrx = dky * zr - dkz * yr;
      real dkry = dkz * xr - dkx * zr;
      real dkrz = dkx * yr - dky * xr;
      real dikx = diy * dkz - diz * dky;
      real diky = diz * dkx - dix * dkz;
      real dikz = dix * dky - diy * dkx;

      real qirx = qiz * yr - qiy * zr;
      real qiry = qix * zr - qiz * xr;
      real qirz = qiy * xr - qix * yr;
      real qkrx = qkz * yr - qky * zr;
      real qkry = qkx * zr - qkz * xr;
      real qkrz = qky * xr - qkx * yr;
      real qikx = qky * qiz - qkz * qiy;
      real qiky = qkz * qix - qkx * qiz;
      real qikz = qkx * qiy - qky * qix;

      real qikrx = qizk * yr - qiyk * zr;
      real qikry = qixk * zr - qizk * xr;
      real qikrz = qiyk * xr - qixk * yr;
      real qkirx = qkzi * yr - qkyi * zr;
      real qkiry = qkxi * zr - qkzi * xr;
      real qkirz = qkyi * xr - qkxi * yr;

      real diqkrx = diqkz * yr - diqky * zr;
      real diqkry = diqkx * zr - diqkz * xr;
      real diqkrz = diqky * xr - diqkx * yr;
      real dkqirx = dkqiz * yr - dkqiy * zr;
      real dkqiry = dkqix * zr - dkqiz * xr;
      real dkqirz = dkqiy * xr - dkqix * yr;

      real dqikx = diy * qkz - diz * qky + dky * qiz - dkz * qiy - 2 * (qixy * qkxz + qiyy * qkyz + qiyz * qkzz - qixz * qkxy - qiyz * qkyy - qizz * qkyz);
      real dqiky = diz * qkx - dix * qkz + dkz * qix - dkx * qiz - 2 * (qixz * qkxx + qiyz * qkxy + qizz * qkxz - qixx * qkxz - qixy * qkyz - qixz * qkzz);
      real dqikz = dix * qky - diy * qkx + dkx * qiy - dky * qix - 2 * (qixx * qkxy + qixy * qkyy + qixz * qkyz - qixy * qkxx - qiyy * qkxy - qiyz * qkxz);

      pgrad.ttmi[0] = -rr3ik * dikx + term1 * dirx + term3 * (dqikx + dkqirx) - term4 * qirx - term6 * (qikrx + qikx);
      pgrad.ttmi[1] = -rr3ik * diky + term1 * diry + term3 * (dqiky + dkqiry) - term4 * qiry - term6 * (qikry + qiky);
      pgrad.ttmi[2] = -rr3ik * dikz + term1 * dirz + term3 * (dqikz + dkqirz) - term4 * qirz - term6 * (qikrz + qikz);
      pgrad.ttmk[0] = rr3ik * dikx + term2 * dkrx - term3 * (dqikx + diqkrx) - term5 * qkrx - term6 * (qkirx - qikx);
      pgrad.ttmk[1] = rr3ik * diky + term2 * dkry - term3 * (dqiky + diqkry) - term5 * qkry - term6 * (qkiry - qiky);
      pgrad.ttmk[2] = rr3ik * dikz + term2 * dkrz - term3 * (dqikz + diqkrz) - term5 * qkrz - term6 * (qkirz - qikz);
   } // end if (do_g)
}
}
