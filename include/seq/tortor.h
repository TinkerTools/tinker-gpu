#pragma once
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"
#include <tinker/detail/ktrtor.hh>

namespace tinker {
// see also bicubic.f
#define BCUCOF__                                                                                                                                              \
   real c[4][4];                                                                                                                                              \
   real d1 = x1u - x1l;                                                                                                                                       \
   real d2 = x2u - x2l;                                                                                                                                       \
   real d12 = d1 * d2;                                                                                                                                        \
   real y1[4], y2[4], y12[4];                                                                                                                                 \
                                                                                                                                                              \
   y1[0] = d1 * y1i[0];                                                                                                                                       \
   y1[1] = d1 * y1i[1];                                                                                                                                       \
   y1[2] = d1 * y1i[2];                                                                                                                                       \
   y1[3] = d1 * y1i[3];                                                                                                                                       \
   y2[0] = d2 * y2i[0];                                                                                                                                       \
   y2[1] = d2 * y2i[1];                                                                                                                                       \
   y2[2] = d2 * y2i[2];                                                                                                                                       \
   y2[3] = d2 * y2i[3];                                                                                                                                       \
   y12[0] = d12 * y12i[0];                                                                                                                                    \
   y12[1] = d12 * y12i[1];                                                                                                                                    \
   y12[2] = d12 * y12i[2];                                                                                                                                    \
   y12[3] = d12 * y12i[3];                                                                                                                                    \
                                                                                                                                                              \
   c[0][0] = y[0];                                                                                                                                            \
   c[0][1] = y2[0];                                                                                                                                           \
   c[0][2] = 3 * (y[3] - y[0]) - (2 * y2[0] + y2[3]);                                                                                                         \
   c[0][3] = 2 * (y[0] - y[3]) + y2[0] + y2[3];                                                                                                               \
   c[1][0] = y1[0];                                                                                                                                           \
   c[1][1] = y12[0];                                                                                                                                          \
   c[1][2] = 3 * (y1[3] - y1[0]) - (2 * y12[0] + y12[3]);                                                                                                     \
   c[1][3] = 2 * (y1[0] - y1[3]) + y12[0] + y12[3];                                                                                                           \
   c[2][0] = 3 * (y[1] - y[0]) - (2 * y1[0] + y1[1]);                                                                                                         \
   c[2][1] = 3 * (y2[1] - y2[0]) - (2 * y12[0] + y12[1]);                                                                                                     \
   c[2][2] = 9 * (y[0] - y[1] + y[2] - y[3]) + 6 * y1[0] + 3 * y1[1] - 3 * y1[2] - 6 * y1[3] + 6 * y2[0] - 6 * y2[1] - 3 * y2[2] + 3 * y2[3] + 4 * y12[0]     \
      + 2 * y12[1] + y12[2] + 2 * y12[3];                                                                                                                     \
   c[2][3] = 6 * (y[1] - y[0] + y[3] - y[2]) + -4 * y1[0] - 2 * y1[1] + 2 * y1[2] + 4 * y1[3] + -3 * y2[0] + 3 * y2[1] + 3 * y2[2] - 3 * y2[3] - 2 * y12[0]   \
      - y12[1] - y12[2] - 2 * y12[3];                                                                                                                         \
   c[3][0] = 2 * (y[0] - y[1]) + y1[0] + y1[1];                                                                                                               \
   c[3][1] = 2 * (y2[0] - y2[1]) + y12[0] + y12[1];                                                                                                           \
   c[3][2] = 6 * (y[1] - y[0] + y[3] - y[2]) + 3 * (y1[2] + y1[3] - y1[0] - y1[1]) + 2 * (2 * (y2[1] - y2[0]) + y2[2] - y2[3]) + -2 * (y12[0] + y12[1])       \
      - y12[2] - y12[3];                                                                                                                                      \
   c[3][3] = 4 * (y[0] - y[1] + y[2] - y[3]) + 2 * (y1[0] + y1[1] - y1[2] - y1[3]) + 2 * (y2[0] - y2[1] - y2[2] + y2[3]) + y12[0] + y12[1] + y12[2] + y12[3]; \
                                                                                                                                                              \
   real t = (x1 - x1l) * REAL_RECIP(x1u - x1l);                                                                                                               \
   real u = (x2 - x2l) * REAL_RECIP(x2u - x2l)

#define BCUCOF_ANSY__                                                   \
   real ay = ((c[3][3] * u + c[3][2]) * u + c[3][1]) * u + c[3][0];     \
   ay = t * ay + ((c[2][3] * u + c[2][2]) * u + c[2][1]) * u + c[2][0]; \
   ay = t * ay + ((c[1][3] * u + c[1][2]) * u + c[1][1]) * u + c[1][0]; \
   ay = t * ay + ((c[0][3] * u + c[0][2]) * u + c[0][1]) * u + c[0][0]

SEQ_ROUTINE
static void bcuint0(const real (&restrict y)[4],
                    const real (&restrict y1i)[4],
                    const real (&restrict y2i)[4],
                    const real (&restrict y12i)[4],
                    real x1l,
                    real x1u,
                    real x2l,
                    real x2u,
                    real x1,
                    real x2,
                    real& restrict ansy)
{
   BCUCOF__;
   BCUCOF_ANSY__;
   ansy = ay;
}

SEQ_ROUTINE
static void bcuint1(const real (&restrict y)[4],
                    const real (&restrict y1i)[4],
                    const real (&restrict y2i)[4],
                    const real (&restrict y12i)[4],
                    real x1l,
                    real x1u,
                    real x2l,
                    real x2u,
                    real x1,
                    real x2,
                    real& restrict ansy,
                    real& restrict ansy1,
                    real& restrict ansy2)
{
   BCUCOF__;
   BCUCOF_ANSY__;
   ansy = ay;

   real day1 = (3 * c[3][3] * t + 2 * c[2][3]) * t + c[1][3];
   day1 = u * day1 + (3 * c[3][2] * t + 2 * c[2][2]) * t + c[1][2];
   day1 = u * day1 + (3 * c[3][1] * t + 2 * c[2][1]) * t + c[1][1];
   day1 = u * day1 + (3 * c[3][0] * t + 2 * c[2][0]) * t + c[1][0];
   day1 *= REAL_RECIP(x1u - x1l);
   ansy1 = day1;

   real day2 = (3 * c[3][3] * u + 2 * c[3][2]) * u + c[3][1];
   day2 = t * day2 + (3 * c[2][3] * u + 2 * c[2][2]) * u + c[2][1];
   day2 = t * day2 + (3 * c[1][3] * u + 2 * c[1][2]) * u + c[1][1];
   day2 = t * day2 + (3 * c[0][3] * u + 2 * c[0][2]) * u + c[0][1];
   day2 *= REAL_RECIP(x2u - x2l);
   ansy2 = day2;
}
#undef BCUCOF__
#undef BCUCOF_ANSY__

#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_tortor(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict dettx,
               grad_prec* restrict detty,
               grad_prec* restrict dettz,

               real ttorunit,
               int itortor,
               const int (*restrict itt)[3],
               const int (*restrict ibitor)[5],
               const int* restrict chkttor_ia_,

               const int* restrict tnx,
               const int* restrict tny,
               const real (*restrict ttx)[ktrtor::maxtgrd],
               const real (*restrict tty)[ktrtor::maxtgrd],
               const real (*restrict tbf)[ktrtor::maxtgrd2],
               const real (*restrict tbx)[ktrtor::maxtgrd2],
               const real (*restrict tby)[ktrtor::maxtgrd2],
               const real (*restrict tbxy)[ktrtor::maxtgrd2],

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

   real ftt[4], ft12[4], ft1[4], ft2[4];

   int i = itt[itortor][0];
   int k = itt[itortor][1]; // parameter indice
   int ia, ib, ic, id, ie;
   // if (itt(3,itortor) .eq. 1) then
   // else
   // end if
   if (itt[itortor][2] == 0) {
      ia = ibitor[i][0];
      ib = ibitor[i][1];
      ic = ibitor[i][2];
      id = ibitor[i][3];
      ie = ibitor[i][4];
   } else {
      ia = ibitor[i][4];
      ib = ibitor[i][3];
      ic = ibitor[i][2];
      id = ibitor[i][1];
      ie = ibitor[i][0];
   }

   // compute the values of the torsional angles

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
   real xie = x[ie];
   real yie = y[ie];
   real zie = z[ie];
   real xba = xib - xia;
   real yba = yib - yia;
   real zba = zib - zia;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;
   real xdc = xid - xic;
   real ydc = yid - yic;
   real zdc = zid - zic;
   real xed = xie - xid;
   real yed = yie - yid;
   real zed = zie - zid;

   real xt = yba * zcb - ycb * zba;
   real yt = zba * xcb - zcb * xba;
   real zt = xba * ycb - xcb * yba;
   real xu = ycb * zdc - ydc * zcb;
   real yu = zcb * xdc - zdc * xcb;
   real zu = xcb * ydc - xdc * ycb;
   real rt2 = xt * xt + yt * yt + zt * zt;
   real ru2 = xu * xu + yu * yu + zu * zu;
   real rtru = REAL_SQRT(rt2 * ru2);
   real xv = ydc * zed - yed * zdc;
   real yv = zdc * xed - zed * xdc;
   real zv = xdc * yed - xed * ydc;
   real rv2 = xv * xv + yv * yv + zv * zv;
   real rurv = REAL_SQRT(ru2 * rv2);

   if (rtru != 0 && rurv != 0) {
      real sign;

      real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);
      real cosine1 = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
      cosine1 = REAL_MIN((real)1, REAL_MAX((real)-1, cosine1));
      sign = xba * xu + yba * yu + zba * zu;
      real angle1 = (sign < 0 ? -radian : radian) * REAL_ACOS(cosine1);
      real value1 = angle1;

      real rdc = REAL_SQRT(xdc * xdc + ydc * ydc + zdc * zdc);
      real cosine2 = (xu * xv + yu * yv + zu * zv) * REAL_RECIP(rurv);
      cosine2 = REAL_MIN((real)1, REAL_MAX((real)-1, cosine2));
      sign = xcb * xv + ycb * yv + zcb * zv;
      real angle2 = (sign < 0 ? -radian : radian) * REAL_ACOS(cosine2);
      real value2 = angle2;

      // check for inverted chirality at the central atom

      const int chk_ia = chkttor_ia_[itortor];
      real vol = 1;
      if (chk_ia >= 0) {
         // if chiral
         real xac = x[chk_ia] - x[ic];
         real yac = y[chk_ia] - y[ic];
         real zac = z[chk_ia] - z[ic];
         real c1 = -ycb * zdc + zcb * ydc;
         real c2 = ydc * zac - zdc * yac;
         real c3 = -yac * zcb + zac * ycb;
         vol = xac * c1 - xcb * c2 + xdc * c3;
      }
      /*
       *          | non-chiral | chiral
       * vol < 0  | sign = 1   | sign = -1; flip values
       * vol >= 0 | sign = 1   | sign = 1
       */
      sign = (vol < 0 ? -1 : 1);
      value1 = (vol < 0 ? -value1 : value1);
      value2 = (vol < 0 ? -value2 : value2);
      // angles must be within the range [-180, +180); +180 is not allowed
      if (value1 < -180)
         value1 += 360;
      if (value1 >= 180)
         value1 -= 360;
      if (value2 < -180)
         value2 += 360;
      if (value2 >= 180)
         value2 -= 360;

      // use bicubic interpolation to compute spline values

      // e.g., -180, -165, -150, ..., 165, 180, tnx[k] = 25
      // xlo = floor((value+180) / (360/(25-1)))
      int tnxk = tnx[k];
      int xlo = REAL_FLOOR((value1 + 180) * (tnxk - 1) * REAL_RECIP(360));
      int ylo = REAL_FLOOR((value2 + 180) * (tny[k] - 1) * REAL_RECIP(360));
      real x1l = ttx[k][xlo];
      real x1u = ttx[k][xlo + 1];
      real y1l = tty[k][ylo];
      real y1u = tty[k][ylo + 1];

      int pos2 = (ylo + 1) * tnxk + xlo;
      int pos1 = pos2 - tnxk;
      ftt[0] = tbf[k][pos1];
      ftt[1] = tbf[k][pos1 + 1];
      ftt[2] = tbf[k][pos2 + 1];
      ftt[3] = tbf[k][pos2];
      ft1[0] = tbx[k][pos1];
      ft1[1] = tbx[k][pos1 + 1];
      ft1[2] = tbx[k][pos2 + 1];
      ft1[3] = tbx[k][pos2];
      ft2[0] = tby[k][pos1];
      ft2[1] = tby[k][pos1 + 1];
      ft2[2] = tby[k][pos2 + 1];
      ft2[3] = tby[k][pos2];
      ft12[0] = tbxy[k][pos1];
      ft12[1] = tbxy[k][pos1 + 1];
      ft12[2] = tbxy[k][pos2 + 1];
      ft12[3] = tbxy[k][pos2];

      real dedang1, dedang2;
      if CONSTEXPR (do_g) {
         bcuint1(ftt, ft1, ft2, ft12, x1l, x1u, y1l, y1u, value1, value2, e, dedang1, dedang2);
      } else {
         bcuint0(ftt, ft1, ft2, ft12, x1l, x1u, y1l, y1u, value1, value2, e);
      }

      if CONSTEXPR (do_e) {
         e *= ttorunit;
      }

      if CONSTEXPR (do_g) {
         dedang1 *= (sign * ttorunit * radian);
         dedang2 *= (sign * ttorunit * radian);

         // chain rule terms for first angle derivative components

         real xca = xic - xia;
         real yca = yic - yia;
         real zca = zic - zia;
         real xdb = xid - xib;
         real ydb = yid - yib;
         real zdb = zid - zib;

         real rt2cb_inv = REAL_RECIP(rt2 * rcb);
         real ru2cb_inv = REAL_RECIP(ru2 * rcb);
         real dedxt = dedang1 * (yt * zcb - ycb * zt) * rt2cb_inv;
         real dedyt = dedang1 * (zt * xcb - zcb * xt) * rt2cb_inv;
         real dedzt = dedang1 * (xt * ycb - xcb * yt) * rt2cb_inv;
         real dedxu = -dedang1 * (yu * zcb - ycb * zu) * ru2cb_inv;
         real dedyu = -dedang1 * (zu * xcb - zcb * xu) * ru2cb_inv;
         real dedzu = -dedang1 * (xu * ycb - xcb * yu) * ru2cb_inv;

         // compute first derivative components for first angle

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

         // chain rule terms for second angle derivative components

         real xec = xie - xic;
         real yec = yie - yic;
         real zec = zie - zic;

         real ru2dc_inv = REAL_RECIP(ru2 * rdc);
         real rv2dc_inv = REAL_RECIP(rv2 * rdc);
         real dedxu2 = dedang2 * (yu * zdc - ydc * zu) * ru2dc_inv;
         real dedyu2 = dedang2 * (zu * xdc - zdc * xu) * ru2dc_inv;
         real dedzu2 = dedang2 * (xu * ydc - xdc * yu) * ru2dc_inv;
         real dedxv2 = -dedang2 * (yv * zdc - ydc * zv) * rv2dc_inv;
         real dedyv2 = -dedang2 * (zv * xdc - zdc * xv) * rv2dc_inv;
         real dedzv2 = -dedang2 * (xv * ydc - xdc * yv) * rv2dc_inv;

         // compute first derivative components for second angle

         real dedxib2 = zdc * dedyu2 - ydc * dedzu2;
         real dedyib2 = xdc * dedzu2 - zdc * dedxu2;
         real dedzib2 = ydc * dedxu2 - xdc * dedyu2;
         real dedxic2 = ydb * dedzu2 - zdb * dedyu2 + zed * dedyv2 - yed * dedzv2;
         real dedyic2 = zdb * dedxu2 - xdb * dedzu2 + xed * dedzv2 - zed * dedxv2;
         real dedzic2 = xdb * dedyu2 - ydb * dedxu2 + yed * dedxv2 - xed * dedyv2;
         real dedxid2 = zcb * dedyu2 - ycb * dedzu2 + yec * dedzv2 - zec * dedyv2;
         real dedyid2 = xcb * dedzu2 - zcb * dedxu2 + zec * dedxv2 - xec * dedzv2;
         real dedzid2 = ycb * dedxu2 - xcb * dedyu2 + xec * dedyv2 - yec * dedxv2;
         real dedxie2 = zdc * dedyv2 - ydc * dedzv2;
         real dedyie2 = xdc * dedzv2 - zdc * dedxv2;
         real dedzie2 = ydc * dedxv2 - xdc * dedyv2;

         atomic_add(dedxia, dettx, ia);
         atomic_add(dedyia, detty, ia);
         atomic_add(dedzia, dettz, ia);
         atomic_add(dedxib + dedxib2, dettx, ib);
         atomic_add(dedyib + dedyib2, detty, ib);
         atomic_add(dedzib + dedzib2, dettz, ib);
         atomic_add(dedxic + dedxic2, dettx, ic);
         atomic_add(dedyic + dedyic2, detty, ic);
         atomic_add(dedzic + dedzic2, dettz, ic);
         atomic_add(dedxid + dedxid2, dettx, id);
         atomic_add(dedyid + dedyid2, detty, id);
         atomic_add(dedzid + dedzid2, dettz, id);
         atomic_add(dedxie2, dettx, ie);
         atomic_add(dedyie2, detty, ie);
         atomic_add(dedzie2, dettz, ie);

         if CONSTEXPR (do_v) {
            vxx = xcb * (dedxic + dedxid) - xba * dedxia + xdc * dedxid;
            vyx = ycb * (dedxic + dedxid) - yba * dedxia + ydc * dedxid;
            vzx = zcb * (dedxic + dedxid) - zba * dedxia + zdc * dedxid;
            vyy = ycb * (dedyic + dedyid) - yba * dedyia + ydc * dedyid;
            vzy = zcb * (dedyic + dedyid) - zba * dedyia + zdc * dedyid;
            vzz = zcb * (dedzic + dedzid) - zba * dedzia + zdc * dedzid;
            real vxx2 = xdc * (dedxid2 + dedxie2) - xcb * dedxib2 + xed * dedxie2;
            real vyx2 = ydc * (dedxid2 + dedxie2) - ycb * dedxib2 + yed * dedxie2;
            real vzx2 = zdc * (dedxid2 + dedxie2) - zcb * dedxib2 + zed * dedxie2;
            real vyy2 = ydc * (dedyid2 + dedyie2) - ycb * dedyib2 + yed * dedyie2;
            real vzy2 = zdc * (dedyid2 + dedyie2) - zcb * dedyib2 + zed * dedyie2;
            real vzz2 = zdc * (dedzid2 + dedzie2) - zcb * dedzib2 + zed * dedzie2;

            vxx += vxx2, vyx += vyx2, vzx += vzx2, vyy += vyy2, vzy += vzy2, vzz += vzz2;
         }
      }
   }
}
}
