#pragma once
#include "seq/damp_hippo.h"
#include "seq/pair_mpole.h"

namespace tinker {
#pragma acc routine seq
template <bool do_e, bool do_g, class ETYP, int CFLX>
SEQ_CUDA
void pair_mpole_chgpen( //
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

   // chgpen terms
   real term1 = corei * corek;
   real term1i = corek * vali;
   real term2i = corek * dir;
   real term3i = corek * qir;
   real term1k = corei * valk;
   real term2k = -corei * dkr;
   real term3k = corei * qkr;
   real term1ik = vali * valk;
   real term2ik = valk * dir - vali * dkr + dik;
   real term3ik = vali * qkr + valk * qir - dir * dkr + 2 * (dkqi - diqk + qiqk);
   real term4ik = dir * qkr - dkr * qir - 4 * qik;
   real term5ik = qir * qkr;

   real rr1i, rr3i, rr5i, rr7i, rr1k, rr3k, rr5k, rr7k, rr1ik, rr3ik, rr5ik, rr7ik, rr9ik, rr11ik;

   // Compute damping factors
   if CONSTEXPR (do_g) {
      damp_pole_v2<11>(dmpik, dmpi, dmpk, r, alphai, alphak);
   } else {
      damp_pole_v2<9>(dmpik, dmpi, dmpk, r, alphai, alphak);
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

   rr1i = bn[0] - (1 - mscale * dmpi[0]) * rr1;
   rr3i = bn[1] - (1 - mscale * dmpi[1]) * rr3;
   rr5i = bn[2] - (1 - mscale * dmpi[2]) * rr5;
   rr7i = bn[3] - (1 - mscale * dmpi[3]) * rr7;
   rr1k = bn[0] - (1 - mscale * dmpk[0]) * rr1;
   rr3k = bn[1] - (1 - mscale * dmpk[1]) * rr3;
   rr5k = bn[2] - (1 - mscale * dmpk[2]) * rr5;
   rr7k = bn[3] - (1 - mscale * dmpk[3]) * rr7;
   rr1ik = bn[0] - (1 - mscale * dmpik[0]) * rr1;
   rr3ik = bn[1] - (1 - mscale * dmpik[1]) * rr3;
   rr5ik = bn[2] - (1 - mscale * dmpik[2]) * rr5;
   rr7ik = bn[3] - (1 - mscale * dmpik[3]) * rr7;
   rr9ik = bn[4] - (1 - mscale * dmpik[4]) * rr9;
   if CONSTEXPR (do_g)
      rr11ik = bn[5] - (1 - mscale * dmpik[5]) * rr11;
   rr1 = bn[0] - (1 - mscale) * rr1;
   rr3 = bn[1] - (1 - mscale) * rr3;

   if CONSTEXPR (do_e)
      e = term1 * rr1 + term4ik * rr7ik + term5ik * rr9ik + term1i * rr1i + term1k * rr1k + term1ik * rr1ik + term2i * rr3i + term2k * rr3k + term2ik * rr3ik
         + term3i * rr5i + term3k * rr5k + term3ik * rr5ik;
   // end if (do_e)

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

      real de = term1 * rr3 + term4ik * rr9ik + term5ik * rr11ik + term1i * rr3i + term1k * rr3k + term1ik * rr3ik + term2i * rr5i + term2k * rr5k
         + term2ik * rr5ik + term3i * rr7i + term3k * rr7k + term3ik * rr7ik;

      term1 = -corek * rr3i - valk * rr3ik + dkr * rr5ik - qkr * rr7ik;
      real term2 = corei * rr3k + vali * rr3ik + dir * rr5ik + qir * rr7ik;
      real term3 = 2 * rr5ik;
      real term4 = -2 * (corek * rr5i + valk * rr5ik - dkr * rr7ik + qkr * rr9ik);
      real term5 = -2 * (corei * rr5k + vali * rr5ik + dir * rr7ik + qir * rr9ik);
      real term6 = 4 * rr7ik;

      if CONSTEXPR (CFLX) {
         real t1i = corek * rr1i + valk * rr1ik;
         real t1k = corei * rr1k + vali * rr1ik;
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
