#pragma once
#include "elec.h"
#include "md.h"
#include "seq_damp.h"


TINKER_NAMESPACE_BEGIN
#pragma acc routine seq
template <elec_t ETYP>
SEQ_CUDA
void pair_dfield(real r2, real xr, real yr, real zr, real dscale,
                 real pscale, //
                 real ci, real dix, real diy, real diz, real qixx, real qixy,
                 real qixz, real qiyy, real qiyz, real qizz, real pdi,
                 real pti, //
                 real ck, real dkx, real dky, real dkz, real qkxx, real qkxy,
                 real qkxz, real qkyy, real qkyz, real qkzz, real pdk,
                 real ptk, //
                 real aewald, real3& restrict fid, real3& restrict fip,
                 real3& restrict fkd, real3& restrict fkp)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5, scale7;
   damp_thole3(r, pdi, pti, pdk, ptk, scale3, scale5, scale7);

   real bn[4];
   if CONSTEXPR (ETYP == elec_t::ewald)
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

   if CONSTEXPR (ETYP == elec_t::ewald) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
      bn[3] -= (1 - scale7) * rr7;
   } else if CONSTEXPR (ETYP == elec_t::coulomb) {
      bn[1] = dscale * scale3 * rr3;
      bn[2] = dscale * scale5 * rr5;
      bn[3] = dscale * scale7 * rr7;
   }

   c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
   inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;
   fid += inci;

   c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
   inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   fkd += inck;

   // p-field

   if CONSTEXPR (ETYP == elec_t::ewald) {
      fip += inci;
      fkp += inck;
   } else if CONSTEXPR (ETYP == elec_t::coulomb) {
      if (pscale == dscale) {
         fip += inci;
         fkp += inck;
      } else {
         bn[1] = pscale * scale3 * rr3;
         bn[2] = pscale * scale5 * rr5;
         bn[3] = pscale * scale7 * rr7;

         c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
         fip += c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;

         c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
         fkp += c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
      }
   }
}


template <elec_t ETYP>
SEQ_CUDA
void pair_ufield(real r2, real xr, real yr, real zr, real uscale, //
                 real uindi0, real uindi1, real uindi2, real uinpi0,
                 real uinpi1, real uinpi2, real pdi, real pti, //
                 real uindk0, real uindk1, real uindk2, real uinpk0,
                 real uinpk1, real uinpk2, real pdk, real ptk, //
                 real aewald, real3& restrict fid, real3& restrict fip,
                 real3& restrict fkd, real3& restrict fkp)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   if CONSTEXPR (ETYP == elec_t::ewald)
      damp_ewald<3>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   if CONSTEXPR (ETYP == elec_t::ewald) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
   } else if CONSTEXPR (ETYP == elec_t::coulomb) {
      bn[1] = uscale * scale3 * rr3;
      bn[2] = uscale * scale5 * rr5;
   }

   real coef;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 uip = make_real3(uinpi0, uinpi1, uinpi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);
   real3 ukp = make_real3(uinpk0, uinpk1, uinpk2);

   coef = bn[2] * dot3(dr, ukd);
   fid += coef * dr - bn[1] * ukd;

   coef = bn[2] * dot3(dr, ukp);
   fip += coef * dr - bn[1] * ukp;

   coef = bn[2] * dot3(dr, uid);
   fkd += coef * dr - bn[1] * uid;

   coef = bn[2] * dot3(dr, uip);
   fkp += coef * dr - bn[1] * uip;
}
TINKER_NAMESPACE_END
