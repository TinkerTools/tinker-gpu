#pragma once
#include "ff/elec.h"
#include "math/realn.h"
#include "seq/add.h"
#include "seq/damp.h"
#include "seq/dampaplus.h"

namespace tinker {
#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_dfield_aplus(real r2,
                       real xr,
                       real yr,
                       real zr,
                       real dscale,
                       real ci,
                       real dix,
                       real diy,
                       real diz,
                       real pdi,
                       real ddi,
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
                       real pdk,
                       real ddk,
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

   real scale3, scale5, scale7;
   damp_aplus3(r, pdi, ddi, pdk, ddk, scale3, scale5, scale7);

   real bn[4];
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

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
      bn[3] -= (1 - scale7) * rr7;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = dscale * scale3 * rr3;
      bn[2] = dscale * scale5 * rr5;
      bn[3] = dscale * scale7 * rr7;
   }

   c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
   inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
   inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_dfield_aplus_v2(real r2,
                          real xr,
                          real yr,
                          real zr,
                          real dscale,
                          real ci,
                          real dix,
                          real diy,
                          real diz,
                          real pdi,
                          real ddi,
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
                          real pdk,
                          real ddk,
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

   real scale3, scale5, scale7;
   damp_aplus3(r, pdi, ddi, pdk, ddk, scale3, scale5, scale7);

   real bn[4];
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
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - dscale * scale3) * rr3;
      bn[2] -= (1 - dscale * scale5) * rr5;
      bn[3] -= (1 - dscale * scale7) * rr7;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = dscale * scale3 * rr3;
      bn[2] = dscale * scale5 * rr5;
      bn[3] = dscale * scale7 * rr7;
   }

   c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
   inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
   inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_ufield_aplus(real r2,
                       real xr,
                       real yr,
                       real zr,
                       real uscale, //
                       real uindi0,
                       real uindi1,
                       real uindi2, //
                       real pdi,
                       real pti, //
                       real uindk0,
                       real uindk1,
                       real uindk2, //
                       real pdk,
                       real ptk, //
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

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   if CONSTEXPR (eq<ETYP, EWALD>())
      damp_ewald<3>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = uscale * scale3 * rr3;
      bn[2] = uscale * scale5 * rr5;
   }

   real coef;
   real3 inci, inck;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);

   coef = bn[2] * dot3(dr, ukd);
   inci = coef * dr - bn[1] * ukd;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   coef = bn[2] * dot3(dr, uid);
   inck = coef * dr - bn[1] * uid;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_ufield_aplus_v2(real r2,
                          real xr,
                          real yr,
                          real zr,
                          real uscale, //
                          real uindi0,
                          real uindi1,
                          real uindi2, //
                          real pdi,
                          real pti, //
                          real uindk0,
                          real uindk1,
                          real uindk2, //
                          real pdk,
                          real ptk, //
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

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   if CONSTEXPR (eq<ETYP, EWALD>())
      damp_ewald<3>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - uscale * scale3) * rr3;
      bn[2] -= (1 - uscale * scale5) * rr5;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = uscale * scale3 * rr3;
      bn[2] = uscale * scale5 * rr5;
   }

   real coef;
   real3 inci, inck;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);

   coef = bn[2] * dot3(dr, ukd);
   inci = coef * dr - bn[1] * ukd;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   coef = bn[2] * dot3(dr, uid);
   inck = coef * dr - bn[1] * uid;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;
}
}
