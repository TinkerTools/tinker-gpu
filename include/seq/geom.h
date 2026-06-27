#pragma once
#include "ff/image.h"
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_geom_group(real& restrict e,
                   real& restrict vxx,
                   real& restrict vyx,
                   real& restrict vzx,
                   real& restrict vyy,
                   real& restrict vzy,
                   real& restrict vzz,

                   grad_prec* restrict degx,
                   grad_prec* restrict degy,
                   grad_prec* restrict degz,

                   int i,
                   const int (*restrict igfix)[2],
                   const real (*restrict gfix)[3],

                   const real* restrict x,
                   const real* restrict y,
                   const real* restrict z,
                   const double* restrict mass,
                   const int* restrict molec,
                   const int (*restrict igrp)[2],
                   const int* restrict kgrp,
                   const double* restrict grpmass,
                   TINKER_IMAGE_PARAMS)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   int ia = igfix[i][0];
   int ib = igfix[i][1];
   int ja1 = igrp[ia][0];
   int ja2 = igrp[ia][1];
   int jb1 = igrp[ib][0];
   int jb2 = igrp[ib][1];

   // image() calls are necessary here because atoms in the same group
   // do not necessarily belong to the same molecule

   real weigha = REAL_MAX((real)1, grpmass[ia]);
   int ka1 = kgrp[ja1];
   real xa1 = x[ka1], ya1 = y[ka1], za1 = z[ka1];
   real xacm = xa1 * weigha, yacm = ya1 * weigha, zacm = za1 * weigha;
   #pragma acc loop seq
   for (int j = ja1 + 1; j < ja2; ++j) {
      int k = kgrp[j];
      real weigh = mass[k];
      real xr = x[k] - xa1;
      real yr = y[k] - ya1;
      real zr = z[k] - za1;
      if (molec[ka1] != molec[k])
         image(xr, yr, zr);
      xacm += xr * weigh;
      yacm += yr * weigh;
      zacm += zr * weigh;
   }
   weigha = REAL_RECIP(weigha);

   real weighb = REAL_MAX((real)1, grpmass[ib]);
   int kb1 = kgrp[jb1];
   real xb1 = x[kb1], yb1 = y[kb1], zb1 = z[kb1];
   real xbcm = xb1 * weighb, ybcm = yb1 * weighb, zbcm = zb1 * weighb;
   #pragma acc loop seq
   for (int j = jb1 + 1; j < jb2; ++j) {
      int k = kgrp[j];
      real weigh = mass[k];
      real xr = x[k] - xb1;
      real yr = y[k] - yb1;
      real zr = z[k] - zb1;
      if (molec[kb1] != molec[k])
         image(xr, yr, zr);
      xbcm += xr * weigh;
      ybcm += yr * weigh;
      zbcm += zr * weigh;
   }
   weighb = REAL_RECIP(weighb);

   real xr = xacm * weigha - xbcm * weighb;
   real yr = yacm * weigha - ybcm * weighb;
   real zr = zacm * weigha - zbcm * weighb;

   bool intermol = molec[kgrp[ja1]] != molec[kgrp[jb1]];
   if (intermol)
      image(xr, yr, zr);

   real r = REAL_SQRT(xr * xr + yr * yr + zr * zr);
   real force = gfix[i][0];
   real gf1 = gfix[i][1];
   real gf2 = gfix[i][2];
   real target = (r < gf1 ? gf1 : (r > gf2 ? gf2 : r));
   real dt = r - target;

   if CONSTEXPR (do_e) {
      real dt2 = dt * dt;
      e = force * dt2;
   }
   if CONSTEXPR (do_g) {
      real rinv = (r == 0 ? 1 : REAL_RECIP(r));
      real de = 2 * force * dt * rinv;
      real dedx = de * xr;
      real dedy = de * yr;
      real dedz = de * zr;

      #pragma acc loop seq
      for (int j = ja1; j < ja2; ++j) {
         int k = kgrp[j];
         real ratio = mass[k] * weigha;
         atomic_add(dedx * ratio, degx, k);
         atomic_add(dedy * ratio, degy, k);
         atomic_add(dedz * ratio, degz, k);
      }
      #pragma acc loop seq
      for (int j = jb1; j < jb2; ++j) {
         int k = kgrp[j];
         real ratio = mass[k] * weighb;
         atomic_add(-dedx * ratio, degx, k);
         atomic_add(-dedy * ratio, degy, k);
         atomic_add(-dedz * ratio, degz, k);
      }
      if CONSTEXPR (do_v) {
         vxx = xr * dedx;
         vyx = yr * dedx;
         vzx = zr * dedx;
         vyy = yr * dedy;
         vzy = zr * dedy;
         vzz = zr * dedz;
      }
   }
}

#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_geom_distance(real& restrict e,
                      real& restrict vxx,
                      real& restrict vyx,
                      real& restrict vzx,
                      real& restrict vyy,
                      real& restrict vzy,
                      real& restrict vzz,

                      grad_prec* restrict degx,
                      grad_prec* restrict degy,
                      grad_prec* restrict degz,

                      int i,
                      const int (*restrict idfix)[2],
                      const real (*restrict dfix)[3],

                      const real* restrict x,
                      const real* restrict y,
                      const real* restrict z,
                      const int* restrict molec,
                      TINKER_IMAGE_PARAMS)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   int ia = idfix[i][0];
   int ib = idfix[i][1];
   real force = dfix[i][0];
   real df1 = dfix[i][1];
   real df2 = dfix[i][2];

   real xr = x[ia] - x[ib];
   real yr = y[ia] - y[ib];
   real zr = z[ia] - z[ib];
   bool intermol = molec[ia] != molec[ib];
   if (intermol)
      image(xr, yr, zr);
   real r = REAL_SQRT(xr * xr + yr * yr + zr * zr);
   real target = r;
   if (r < df1)
      target = df1;
   if (r > df2)
      target = df2;
   real dt = r - target;

   if CONSTEXPR (do_e) {
      real dt2 = dt * dt;
      e = force * dt2;
   }
   if CONSTEXPR (do_g) {
      real rinv = (r == 0 ? 1 : REAL_RECIP(r));
      real de = 2 * force * dt * rinv;
      real dedx = de * xr;
      real dedy = de * yr;
      real dedz = de * zr;
      atomic_add(dedx, degx, ia);
      atomic_add(dedy, degy, ia);
      atomic_add(dedz, degz, ia);
      atomic_add(-dedx, degx, ib);
      atomic_add(-dedy, degy, ib);
      atomic_add(-dedz, degz, ib);
      if CONSTEXPR (do_v) {
         vxx = xr * dedx;
         vyx = yr * dedx;
         vzx = zr * dedx;
         vyy = yr * dedy;
         vzy = zr * dedy;
         vzz = zr * dedz;
      }
   }
}

#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_geom_angle(real& restrict e,
                   real& restrict vxx,
                   real& restrict vyx,
                   real& restrict vzx,
                   real& restrict vyy,
                   real& restrict vzy,
                   real& restrict vzz,

                   grad_prec* restrict degx,
                   grad_prec* restrict degy,
                   grad_prec* restrict degz,

                   int i,
                   const int (*restrict iafix)[3],
                   const real (*restrict afix)[3],

                   const real* restrict x,
                   const real* restrict y,
                   const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   if CONSTEXPR (do_e)
      e = 0;
   if CONSTEXPR (do_v)
      vxx = 0, vyx = 0, vzx = 0, vyy = 0, vzy = 0, vzz = 0;

   int ia = iafix[i][0];
   int ib = iafix[i][1];
   int ic = iafix[i][2];
   real force = afix[i][0];
   real af1 = afix[i][1];
   real af2 = afix[i][2];

   real xia = x[ia];
   real yia = y[ia];
   real zia = z[ia];
   real xib = x[ib];
   real yib = y[ib];
   real zib = z[ib];
   real xic = x[ic];
   real yic = y[ic];
   real zic = z[ic];

   real xab = xia - xib;
   real yab = yia - yib;
   real zab = zia - zib;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;
   real rab2 = xab * xab + yab * yab + zab * zab;
   rab2 = REAL_MAX(rab2, (real)0.0001);
   real rcb2 = xcb * xcb + ycb * ycb + zcb * zcb;
   rcb2 = REAL_MAX(rcb2, (real)0.0001);

   real xp = ycb * zab - zcb * yab;
   real yp = zcb * xab - xcb * zab;
   real zp = xcb * yab - ycb * xab;
   real rp = REAL_SQRT(xp * xp + yp * yp + zp * zp);
   rp = REAL_MAX(rp, (real)0.0001);
   real dot = xab * xcb + yab * ycb + zab * zcb;
   real cosine = dot * REAL_RSQRT(rab2 * rcb2);
   cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
   real angle = radian * REAL_ACOS(cosine);

   real target = angle;
   if (angle < af1)
      target = af1;
   if (angle > af2)
      target = af2;
   real dt = (angle - target) * _1radian;

   if CONSTEXPR (do_e) {
      real dt2 = dt * dt;
      e = force * dt2;
   }
   if CONSTEXPR (do_g) {
      real deddt = 2 * force * dt;
      real terma = -deddt * REAL_RECIP(rab2 * rp);
      real termc = deddt * REAL_RECIP(rcb2 * rp);
      real dedxia = terma * (yab * zp - zab * yp);
      real dedyia = terma * (zab * xp - xab * zp);
      real dedzia = terma * (xab * yp - yab * xp);
      real dedxic = termc * (ycb * zp - zcb * yp);
      real dedyic = termc * (zcb * xp - xcb * zp);
      real dedzic = termc * (xcb * yp - ycb * xp);
      real dedxib = -dedxia - dedxic;
      real dedyib = -dedyia - dedyic;
      real dedzib = -dedzia - dedzic;

      atomic_add(dedxia, degx, ia);
      atomic_add(dedyia, degy, ia);
      atomic_add(dedzia, degz, ia);
      atomic_add(dedxib, degx, ib);
      atomic_add(dedyib, degy, ib);
      atomic_add(dedzib, degz, ib);
      atomic_add(dedxic, degx, ic);
      atomic_add(dedyic, degy, ic);
      atomic_add(dedzic, degz, ic);
      if CONSTEXPR (do_v) {
         vxx = xab * dedxia + xcb * dedxic;
         vyx = yab * dedxia + ycb * dedxic;
         vzx = zab * dedxia + zcb * dedxic;
         vyy = yab * dedyia + ycb * dedyic;
         vzy = zab * dedyia + zcb * dedyic;
         vzz = zab * dedzia + zcb * dedzic;
      }
   }
}

#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_geom_torsion(real& restrict e,
                     real& restrict vxx,
                     real& restrict vyx,
                     real& restrict vzx,
                     real& restrict vyy,
                     real& restrict vzy,
                     real& restrict vzz,

                     grad_prec* restrict degx,
                     grad_prec* restrict degy,
                     grad_prec* restrict degz,

                     int i,
                     const int (*restrict itfix)[4],
                     const real (*restrict tfix)[3],

                     const real* restrict x,
                     const real* restrict y,
                     const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   if CONSTEXPR (do_e)
      e = 0;
   if CONSTEXPR (do_v)
      vxx = 0, vyx = 0, vzx = 0, vyy = 0, vzy = 0, vzz = 0;

   int ia = itfix[i][0];
   int ib = itfix[i][1];
   int ic = itfix[i][2];
   int id = itfix[i][3];
   real force = tfix[i][0];
   real tf1 = tfix[i][1];
   real tf2 = tfix[i][2];

   real xia = x[ia];
   real yia = y[ia];
   real zia = z[ia];
   real xib = x[ib];
   real yib = y[ib];
   real zib = z[ib];
   real xic = x[ic];
   real yic = y[ic];
   real zic = z[ic];
   real xid = x[id];
   real yid = y[id];
   real zid = z[id];
   real xba = xib - xia;
   real yba = yib - yia;
   real zba = zib - zia;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;
   real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);
   rcb = REAL_MAX(rcb, (real)0.0001);
   real xdc = xid - xic;
   real ydc = yid - yic;
   real zdc = zid - zic;

   real xt = yba * zcb - ycb * zba;
   real yt = zba * xcb - zcb * xba;
   real zt = xba * ycb - xcb * yba;
   real xu = ycb * zdc - ydc * zcb;
   real yu = zcb * xdc - zdc * xcb;
   real zu = xcb * ydc - xdc * ycb;
   real xtu = yt * zu - yu * zt;
   real ytu = zt * xu - zu * xt;
   real ztu = xt * yu - xu * yt;
   real rt2 = xt * xt + yt * yt + zt * zt;
   rt2 = REAL_MAX(rt2, (real)0.0001);
   real ru2 = xu * xu + yu * yu + zu * zu;
   ru2 = REAL_MAX(ru2, (real)0.0001);
   real rtru = REAL_SQRT(rt2 * ru2);

   real cosine = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
   real sine = (xcb * xtu + ycb * ytu + zcb * ztu) * REAL_RECIP(rcb * rtru);
   cosine = REAL_MIN((real)1.0, REAL_MAX((real)-1.0, cosine));
   real angle = radian * REAL_ACOS(cosine);
   if (sine < 0)
      angle = -angle;
   real target;
   if (angle > tf1 and angle < tf2)
      target = angle;
   else if (angle > tf1 and tf1 > tf2)
      target = angle;
   else if (angle < tf2 and tf1 > tf2)
      target = angle;
   else {
      real t1 = angle - tf1;
      real t2 = angle - tf2;
      if (t1 > 180)
         t1 -= 360;
      if (t1 < -180)
         t1 += 360;
      if (t2 > 180)
         t2 -= 360;
      if (t2 < -180)
         t2 += 360;
      if (REAL_ABS(t1) < REAL_ABS(t2))
         target = tf1;
      else
         target = tf2;
   }
   real dt = angle - target;
   if (dt > 180)
      dt -= 360;
   if (dt < -180)
      dt += 360;

   dt *= _1radian;
   if CONSTEXPR (do_e) {
      real dt2 = dt * dt;
      e = force * dt2;
   }
   if CONSTEXPR (do_g) {
      real dedphi = 2 * force * dt;

      real xca = xic - xia;
      real yca = yic - yia;
      real zca = zic - zia;
      real xdb = xid - xib;
      real ydb = yid - yib;
      real zdb = zid - zib;

      real rt_inv = REAL_RECIP(rt2 * rcb);
      real ru_inv = REAL_RECIP(ru2 * rcb);
      real dedxt = dedphi * (yt * zcb - ycb * zt) * rt_inv;
      real dedyt = dedphi * (zt * xcb - zcb * xt) * rt_inv;
      real dedzt = dedphi * (xt * ycb - xcb * yt) * rt_inv;
      real dedxu = -dedphi * (yu * zcb - ycb * zu) * ru_inv;
      real dedyu = -dedphi * (zu * xcb - zcb * xu) * ru_inv;
      real dedzu = -dedphi * (xu * ycb - xcb * yu) * ru_inv;

      real dedxia = zcb * dedyt - ycb * dedzt;
      real dedyia = xcb * dedzt - zcb * dedxt;
      real dedzia = ycb * dedxt - xcb * dedyt;
      real dedxib = yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
      real dedyib = zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
      real dedzib = xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
      real dedxic = zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
      real dedyic = xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
      real dedzic = yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
      real dedxid = zcb * dedyu - ycb * dedzu;
      real dedyid = xcb * dedzu - zcb * dedxu;
      real dedzid = ycb * dedxu - xcb * dedyu;

      atomic_add(dedxia, degx, ia);
      atomic_add(dedyia, degy, ia);
      atomic_add(dedzia, degz, ia);
      atomic_add(dedxib, degx, ib);
      atomic_add(dedyib, degy, ib);
      atomic_add(dedzib, degz, ib);
      atomic_add(dedxic, degx, ic);
      atomic_add(dedyic, degy, ic);
      atomic_add(dedzic, degz, ic);
      atomic_add(dedxid, degx, id);
      atomic_add(dedyid, degy, id);
      atomic_add(dedzid, degz, id);

      if CONSTEXPR (do_v) {
         vxx = xcb * (dedxic + dedxid) - xba * dedxia + xdc * dedxid;
         vyx = ycb * (dedxic + dedxid) - yba * dedxia + ydc * dedxid;
         vzx = zcb * (dedxic + dedxid) - zba * dedxia + zdc * dedxid;
         vyy = ycb * (dedyic + dedyid) - yba * dedyia + ydc * dedyid;
         vzy = zcb * (dedyic + dedyid) - zba * dedyia + zdc * dedyid;
         vzz = zcb * (dedzic + dedzid) - zba * dedzia + zdc * dedzid;
      }
   }
}

#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_geom_position(real& restrict e,
                      real& restrict vxx,
                      real& restrict vyx,
                      real& restrict vzx,
                      real& restrict vyy,
                      real& restrict vzy,
                      real& restrict vzz,

                      grad_prec* restrict degx,
                      grad_prec* restrict degy,
                      grad_prec* restrict degz,

                      int i,
                      const int* restrict ipfix,
                      const int (*restrict kpfix)[3],
                      const real* restrict xpfix,
                      const real* restrict ypfix,
                      const real* restrict zpfix,
                      const real (*restrict pfix)[2],

                      const real* restrict x,
                      const real* restrict y,
                      const real* restrict z,
                      TINKER_IMAGE_PARAMS)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   int ia = ipfix[i];
   real force = pfix[i][0];
   real radius = pfix[i][1];
   real xr = 0, yr = 0, zr = 0;
   if (kpfix[i][0])
      xr = x[ia] - xpfix[i];
   if (kpfix[i][1])
      yr = y[ia] - ypfix[i];
   if (kpfix[i][2])
      zr = z[ia] - zpfix[i];
   image(xr, yr, zr);
   real r = REAL_SQRT(xr * xr + yr * yr + zr * zr);
   real dt = REAL_MAX((real)0, r - radius);

   if CONSTEXPR (do_e) {
      real dt2 = dt * dt;
      e = force * dt2;
   }
   if CONSTEXPR (do_g) {
      real rinv = (r == 0 ? 1 : REAL_RECIP(r));
      real de = 2 * force * dt * rinv;
      real dedx = de * xr;
      real dedy = de * yr;
      real dedz = de * zr;
      atomic_add(dedx, degx, ia);
      atomic_add(dedy, degy, ia);
      atomic_add(dedz, degz, ia);
      if CONSTEXPR (do_v) {
         vxx = xr * dedx;
         vyx = yr * dedx;
         vzx = zr * dedx;
         vyy = yr * dedy;
         vzy = zr * dedy;
         vzz = zr * dedz;
      }
   }
}
}
