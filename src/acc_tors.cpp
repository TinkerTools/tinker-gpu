#include "add.h"
#include "e_tors.h"
#include "md.h"

// TOOD: use_smooth version

TINKER_NAMESPACE_BEGIN
template <int USE>
void etors_tmpl()
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_v = USE & calc::virial;
   sanity_check<USE>();

   auto bufsize = buffer_size();

   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,gx,gy,gz,\
               itors,tors1,tors2,tors3,tors4,tors5,tors6,\
               et,vir_et)
   for (int i = 0; i < ntors; ++i) {
      int offset = i & (bufsize - 1);
      const int ia = itors[i][0];
      const int ib = itors[i][1];
      const int ic = itors[i][2];
      const int id = itors[i][3];

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
      real ru2 = xu * xu + yu * yu + zu * zu;
      real rtru = REAL_SQRT(rt2 * ru2);

      if (rtru != 0) {
         real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);
         real cosine = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
         real sine =
            (xcb * xtu + ycb * ytu + zcb * ztu) * REAL_RECIP(rcb * rtru);

         // set the torsional parameters for this angle

         real v1 = tors1[i][0];
         real c1 = tors1[i][2];
         real s1 = tors1[i][3];
         real v2 = tors2[i][0];
         real c2 = tors2[i][2];
         real s2 = tors2[i][3];
         real v3 = tors3[i][0];
         real c3 = tors3[i][2];
         real s3 = tors3[i][3];
         real v4 = tors4[i][0];
         real c4 = tors4[i][2];
         real s4 = tors4[i][3];
         real v5 = tors5[i][0];
         real c5 = tors5[i][2];
         real s5 = tors5[i][3];
         real v6 = tors6[i][0];
         real c6 = tors6[i][2];
         real s6 = tors6[i][3];

         // compute the multiple angle trigonometry and the phase terms

         real cosine2 = cosine * cosine - sine * sine;
         real sine2 = 2 * cosine * sine;
         real cosine3 = cosine * cosine2 - sine * sine2;
         real sine3 = cosine * sine2 + sine * cosine2;
         real cosine4 = cosine * cosine3 - sine * sine3;
         real sine4 = cosine * sine3 + sine * cosine3;
         real cosine5 = cosine * cosine4 - sine * sine4;
         real sine5 = cosine * sine4 + sine * cosine4;
         real cosine6 = cosine * cosine5 - sine * sine5;
         real sine6 = cosine * sine5 + sine * cosine5;

         real phi1 = 1 + (cosine * c1 + sine * s1);
         real phi2 = 1 + (cosine2 * c2 + sine2 * s2);
         real phi3 = 1 + (cosine3 * c3 + sine3 * s3);
         real phi4 = 1 + (cosine4 * c4 + sine4 * s4);
         real phi5 = 1 + (cosine5 * c5 + sine5 * s5);
         real phi6 = 1 + (cosine6 * c6 + sine6 * s6);
         real dphi1 = (cosine * s1 - sine * c1);
         real dphi2 = 2 * (cosine2 * s2 - sine2 * c2);
         real dphi3 = 3 * (cosine3 * s3 - sine3 * c3);
         real dphi4 = 4 * (cosine4 * s4 - sine4 * c4);
         real dphi5 = 5 * (cosine5 * s5 - sine5 * c5);
         real dphi6 = 6 * (cosine6 * s6 - sine6 * c6);

         if CONSTEXPR (do_e) {
            real e = torsunit *
               (v1 * phi1 + v2 * phi2 + v3 * phi3 + v4 * phi4 + v5 * phi5 +
                v6 * phi6);
            atomic_add(e, et, offset);
         }

         if CONSTEXPR (do_g) {
            real dedphi = torsunit *
               (v1 * dphi1 + v2 * dphi2 + v3 * dphi3 + v4 * dphi4 + v5 * dphi5 +
                v6 * dphi6);

            // chain rule terms for first derivative components

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

            // compute first derivative components for this angle

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

            atomic_add(dedxia, gx, ia);
            atomic_add(dedyia, gy, ia);
            atomic_add(dedzia, gz, ia);
            atomic_add(dedxib, gx, ib);
            atomic_add(dedyib, gy, ib);
            atomic_add(dedzib, gz, ib);
            atomic_add(dedxic, gx, ic);
            atomic_add(dedyic, gy, ic);
            atomic_add(dedzic, gz, ic);
            atomic_add(dedxid, gx, id);
            atomic_add(dedyid, gy, id);
            atomic_add(dedzid, gz, id);

            if CONSTEXPR (do_v) {
               real vxx = xcb * (dedxic + dedxid) - xba * dedxia + xdc * dedxid;
               real vyx = ycb * (dedxic + dedxid) - yba * dedxia + ydc * dedxid;
               real vzx = zcb * (dedxic + dedxid) - zba * dedxia + zdc * dedxid;
               real vyy = ycb * (dedyic + dedyid) - yba * dedyia + ydc * dedyid;
               real vzy = zcb * (dedyic + dedyid) - zba * dedyia + zdc * dedyid;
               real vzz = zcb * (dedzic + dedzid) - zba * dedzia + zdc * dedzid;

               atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_et, offset);
            }
         }
      }
   } // end for (int i)
}

void etors_acc(int vers)
{
   if (vers == calc::v0 || vers == calc::v3)
      etors_tmpl<calc::v0>();
   else if (vers == calc::v1)
      etors_tmpl<calc::v1>();
   else if (vers == calc::v4)
      etors_tmpl<calc::v4>();
   else if (vers == calc::v5)
      etors_tmpl<calc::v5>();
   else if (vers == calc::v6)
      etors_tmpl<calc::v6>();
}
TINKER_NAMESPACE_END
