#pragma once
#include "ff/elec.h"
#include "math/realn.h"
#include "seq/add.h"
#include "seq/damp.h"
#include "seq/damp_hippo.h"

namespace tinker {
#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_dfield_chgpen(real r2,
                        real xr,
                        real yr,
                        real zr,
                        real dscale,
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
                        real aewald,
                        real& restrict fidx,
                        real& restrict fidy,
                        real& restrict fidz,
                        real& restrict fkdx,
                        real& restrict fkdy,
                        real& restrict fkdz)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real bn[4];
   real dmpi[4], dmpk[4];

   damp_dir(dmpi, dmpk, r, alphai, alphak);

   if CONSTEXPR (eq<ETYP, EWALD>())
      damp_ewald<4>(bn, r, invr1, rr2, aewald);

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;
   real rr7 = 15 * rr1 * rr2 * rr2 * rr2;

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

   real3 dixyz = make_real3(dix, diy, diz);
   real3 dkxyz = make_real3(dkx, dky, dkz);
   real3 qixyz = make_real3(qix, qiy, qiz);
   real3 qkxyz = make_real3(qkx, qky, qkz);
   real3 dr = make_real3(xr, yr, zr);
   real c1;
   real3 inci, inck;

   // d-field
   real rr3i, rr5i, rr7i;
   real rr3k, rr5k, rr7k;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      rr3i = bn[1] - (1 - dscale * dmpi[1]) * rr3;
      rr5i = bn[2] - (1 - dscale * dmpi[2]) * rr5;
      rr7i = bn[3] - (1 - dscale * dmpi[3]) * rr7;
      rr3k = bn[1] - (1 - dscale * dmpk[1]) * rr3;
      rr5k = bn[2] - (1 - dscale * dmpk[2]) * rr5;
      rr7k = bn[3] - (1 - dscale * dmpk[3]) * rr7;
      rr3 = bn[1] - (1 - dscale) * rr3;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      rr3i = dscale * dmpi[1] * rr3;
      rr5i = dscale * dmpi[2] * rr5;
      rr7i = dscale * dmpi[3] * rr7;
      rr3k = dscale * dmpk[1] * rr3;
      rr5k = dscale * dmpk[2] * rr5;
      rr7k = dscale * dmpk[3] * rr7;
      rr3 *= dscale;
   }
   c1 = -(rr3 * corek + rr3k * valk - rr5k * dkr + rr7k * qkr);
   inci = c1 * dr - rr3k * dkxyz + 2 * rr5k * qkxyz;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   c1 = (rr3 * corei + rr3i * vali + rr5i * dir + rr7i * qir);
   inck = c1 * dr - rr3i * dixyz - 2 * rr5i * qixyz;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_ufield_chgpen(real r2,
                        real xr,
                        real yr,
                        real zr,
                        real wscale, //
                        real uindi0,
                        real uindi1,
                        real uindi2, //
                        real corei,
                        real vali,
                        real alphai, //
                        real uindk0,
                        real uindk1,
                        real uindk2, //
                        real corek,
                        real valk,
                        real alphak, //
                        real aewald,
                        real& restrict fidx,
                        real& restrict fidy,
                        real& restrict fidz,
                        real& restrict fkdx,
                        real& restrict fkdy,
                        real& restrict fkdz)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real bn[3];
   real dmpik[3];
   real scale3, scale5;
   damp_mut(dmpik, r, alphai, alphak);
   scale3 = wscale * dmpik[1];
   scale5 = wscale * dmpik[2];

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = rr1 * rr2 * rr2;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      damp_ewald<3>(bn, r, invr1, rr2, aewald);

      bn[1] *= -1;
      bn[1] += (1 - scale3) * rr3;
      bn[2] -= 3 * (1 - scale5) * rr5;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = -scale3 * rr3;
      bn[2] = 3 * scale5 * rr5;
   }

   real coef;
   real3 inci, inck;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);

   coef = bn[2] * dot3(dr, ukd);
   inci = coef * dr + bn[1] * ukd;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   coef = bn[2] * dot3(dr, uid);
   inck = coef * dr + bn[1] * uid;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}
}
