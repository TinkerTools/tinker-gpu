#include "ff/modamoeba.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/torque.h"

namespace tinker {
template <bool DO_V>
__global__
void torque_cu1(int n, VirialBuffer restrict gpu_vir, //
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, const real* restrict x,
   const real* restrict y, const real* restrict z, const LocalFrame* restrict zaxis, const real* restrict trqx,
   const real* restrict trqy, const real* restrict trqz)
{
   real trq[3], frcz[3], frcx[3], frcy[3];
   real deia[3], deib[3], deic[3], deid[3];
   real u[3], v[3], w[3];
   // uv uw: z only, z then x
   // vw:    z only, z then x, bisector
   real uv[3], uw[3], vw[3];
   // z bisector
   real r[3], s[3], ur[3], us[3], vs[3], ws[3], t1[3], t2[3];
   // 3 fold
   real p[3], del[3], eps[3];

   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      auto axetyp = zaxis[i].polaxe;
      if (axetyp == LFRM_NONE)
         continue;

      // get the local frame type and the frame-defining atoms
      auto ia = zaxis[i].zaxis;
      auto ib = i;
      auto ic = zaxis[i].xaxis;
      auto id = INT_ABS(zaxis[i].yaxis) - 1;

      // construct the three rotation axes for the local frame
      // u
      real usiz1;
      u[0] = x[ia] - x[ib];
      u[1] = y[ia] - y[ib];
      u[2] = z[ia] - z[ib];
      usiz1 = REAL_RSQRT(torqueDot(u, u));
      torqueNormal(u, usiz1);

      // v
      real vsiz1;
      if (axetyp != LFRM_Z_ONLY) {
         v[0] = x[ic] - x[ib];
         v[1] = y[ic] - y[ib];
         v[2] = z[ic] - z[ib];
         vsiz1 = REAL_RSQRT(torqueDot(v, v));
      } else {
         int foo = !(REAL_ABS(u[0]) > (real)0.866);
         v[0] = (foo ? 1 : 0);
         v[1] = (foo ? 0 : 1);
         v[2] = 0;
         vsiz1 = 1;
      }
      torqueNormal(v, vsiz1);

      // w = u cross v
      real wsiz1;
      if (axetyp == LFRM_Z_BISECT || axetyp == LFRM_3_FOLD) {
         w[0] = x[id] - x[ib];
         w[1] = y[id] - y[ib];
         w[2] = z[id] - z[ib];
      } else {
         torqueCross(w, u, v);
      }
      wsiz1 = REAL_RSQRT(torqueDot(w, w));
      torqueNormal(w, wsiz1);

      // negative of dot product of torque with unit vectors gives result of
      // infinitesimal rotation along these vectors
      trq[0] = trqx[i];
      trq[1] = trqy[i];
      trq[2] = trqz[i];
      real dphidu = -torqueDot(trq, u);
      real dphidv = -torqueDot(trq, v);
      real dphidw = -torqueDot(trq, w);

      // force distribution
      deia[0] = 0, deia[1] = 0, deia[2] = 0;
      deib[0] = 0, deib[1] = 0, deib[2] = 0;
      deic[0] = 0, deic[1] = 0, deic[2] = 0;
      deid[0] = 0, deid[1] = 0, deid[2] = 0;

      // zero out force components on local frame-defining atoms
      frcz[0] = 0, frcz[1] = 0, frcz[2] = 0;
      frcx[0] = 0, frcx[1] = 0, frcx[2] = 0;
      frcy[0] = 0, frcy[1] = 0, frcy[2] = 0;

      if (axetyp == LFRM_Z_ONLY || axetyp == LFRM_Z_THEN_X || axetyp == LFRM_BISECTOR) {
         // find the perpendicular and angle for each pair of axes
         // v cross u
         real uvsiz1;
         torqueCross(uv, v, u);
         uvsiz1 = REAL_RSQRT(torqueDot(uv, uv));
         torqueNormal(uv, uvsiz1);

         // w cross u
         real uwsiz1;
         torqueCross(uw, w, u);
         uwsiz1 = REAL_RSQRT(torqueDot(uw, uw));
         torqueNormal(uw, uwsiz1);

         // get sine and cosine of angles between the rotation axes
         real uvcos = torqueDot(u, v);
         real uvsin1 = REAL_RSQRT(1 - uvcos * uvcos);

         if (axetyp == LFRM_Z_ONLY) {
            for (int j = 0; j < 3; ++j) {
               real du = uv[j] * dphidv * usiz1 * uvsin1 + uw[j] * dphidw * usiz1;
               deia[j] += du;
               deib[j] -= du;
               if CONSTEXPR (DO_V) {
                  frcz[j] += du;
               }
            }
         } else if (axetyp == LFRM_Z_THEN_X) {
            for (int j = 0; j < 3; ++j) {
               real du = uv[j] * dphidv * usiz1 * uvsin1 + uw[j] * dphidw * usiz1;
               real dv = -uv[j] * dphidu * vsiz1 * uvsin1;
               deia[j] += du;
               deic[j] += dv;
               deib[j] -= (du + dv);
               if CONSTEXPR (DO_V) {
                  frcz[j] += du;
                  frcx[j] += dv;
               }
            }
         } else /* if (axetyp == LFRM_BISECTOR) */ {
            // w cross v
            real vwsiz1;
            torqueCross(vw, w, v);
            vwsiz1 = REAL_RSQRT(torqueDot(vw, vw));
            torqueNormal(vw, vwsiz1);

            for (int j = 0; j < 3; ++j) {
               real du = uv[j] * dphidv * usiz1 * uvsin1 + 0.5f * uw[j] * dphidw * usiz1;
               real dv = -uv[j] * dphidu * vsiz1 * uvsin1 + 0.5f * vw[j] * dphidw * vsiz1;
               deia[j] += du;
               deic[j] += dv;
               deib[j] -= (du + dv);
               if CONSTEXPR (DO_V) {
                  frcz[j] += du;
                  frcx[j] += dv;
               }
            }
         }
      } else if (axetyp == LFRM_Z_BISECT) {
         // build some additional axes needed for the Z-Bisect method
         real rsiz1;
         r[0] = v[0] + w[0];
         r[1] = v[1] + w[1];
         r[2] = v[2] + w[2];
         rsiz1 = REAL_RSQRT(torqueDot(r, r));

         real ssiz1;
         torqueCross(s, u, r);
         ssiz1 = REAL_RSQRT(torqueDot(s, s));

         torqueNormal(r, rsiz1);
         torqueNormal(s, ssiz1);

         // find the perpendicular and angle for each pair of axes
         real ursiz1;
         torqueCross(ur, r, u);
         ursiz1 = REAL_RSQRT(torqueDot(ur, ur));
         torqueNormal(ur, ursiz1);

         real ussiz1;
         torqueCross(us, s, u);
         ussiz1 = REAL_RSQRT(torqueDot(us, us));
         torqueNormal(us, ussiz1);

         real vssiz1;
         torqueCross(vs, s, v);
         vssiz1 = REAL_RSQRT(torqueDot(vs, vs));
         torqueNormal(vs, vssiz1);

         real wssiz1;
         torqueCross(ws, s, w);
         wssiz1 = REAL_RSQRT(torqueDot(ws, ws));
         torqueNormal(ws, wssiz1);

         // get sine and cosine of angles between the rotation axes
         real urcos = torqueDot(u, r);
         real ursin1 = REAL_RSQRT(1 - urcos * urcos);
         real vscos = torqueDot(v, s);
         real vssin = REAL_SQRT(1 - vscos * vscos);
         real wscos = torqueDot(w, s);
         real wssin = REAL_SQRT(1 - wscos * wscos);

         // compute the projection of v and w onto the ru-plane
         real t1siz1;
         t1[0] = v[0] - s[0] * vscos;
         t1[1] = v[1] - s[1] * vscos;
         t1[2] = v[2] - s[2] * vscos;
         t1siz1 = REAL_RSQRT(torqueDot(t1, t1));
         torqueNormal(t1, t1siz1);

         real t2siz1;
         t2[0] = w[0] - s[0] * wscos;
         t2[1] = w[1] - s[1] * wscos;
         t2[2] = w[2] - s[2] * wscos;
         t2siz1 = REAL_RSQRT(torqueDot(t2, t2));
         torqueNormal(t2, t2siz1);

         real ut1cos = torqueDot(u, t1);
         real ut1sin = REAL_SQRT(1 - ut1cos * ut1cos);
         real ut2cos = torqueDot(u, t2);
         real ut2sin = REAL_SQRT(1 - ut2cos * ut2cos);
         real _1_ut1sin_ut2sin = REAL_RECIP(ut1sin + ut2sin);

         // negative of dot product of torque with unit vectors gives result of
         // infinitesimal rotation along these vectors
         real dphidr = -torqueDot(trq, r);
         real dphids = -torqueDot(trq, s);

         for (int j = 0; j < 3; ++j) {
            real du = ur[j] * dphidr * usiz1 * ursin1 + us[j] * dphids * usiz1;
            real dv = (vssin * s[j] - vscos * t1[j]) * dphidu * vsiz1 * _1_ut1sin_ut2sin;
            real dw = (wssin * s[j] - wscos * t2[j]) * dphidu * wsiz1 * _1_ut1sin_ut2sin;
            deia[j] += du;
            deic[j] += dv;
            deid[j] += dw;
            deib[j] -= (du + dv + dw);
            if CONSTEXPR (DO_V) {
               frcz[j] += du;
               frcx[j] += dv;
               frcy[j] += dw;
            }
         }
      } else if (axetyp == LFRM_3_FOLD) {
         real psiz1;
         p[0] = u[0] + v[0] + w[0];
         p[1] = u[1] + v[1] + w[1];
         p[2] = u[2] + v[2] + w[2];
         psiz1 = REAL_RSQRT(torqueDot(p, p));
         torqueNormal(p, psiz1);

         real wpcos = torqueDot(w, p);
         real upcos = torqueDot(u, p);
         real vpcos = torqueDot(v, p);

         real rsiz1;
         real delsiz1;
         real dphidr, dphiddel;

         r[0] = u[0] + v[0];
         r[1] = u[1] + v[1];
         r[2] = u[2] + v[2];
         rsiz1 = REAL_RSQRT(torqueDot(r, r));
         torqueNormal(r, rsiz1);
         real rwcos = torqueDot(r, w);
         real rwsin1 = REAL_RSQRT(1 - rwcos * rwcos);
         dphidr = -torqueDot(trq, r);
         torqueCross(del, r, w);
         delsiz1 = REAL_RSQRT(torqueDot(del, del));
         torqueNormal(del, delsiz1);
         dphiddel = -torqueDot(trq, del);
         torqueCross(eps, del, w);
         for (int j = 0; j < 3; ++j) {
            real dw = del[j] * dphidr * wsiz1 * rwsin1 + eps[j] * dphiddel * wpcos * wsiz1 * psiz1;
            deid[j] += dw;
            deib[j] -= dw;
            if CONSTEXPR (DO_V) {
               frcy[j] += dw;
            }
         }

         r[0] = v[0] + w[0];
         r[1] = v[1] + w[1];
         r[2] = v[2] + w[2];
         rsiz1 = REAL_RSQRT(torqueDot(r, r));
         torqueNormal(r, rsiz1);
         real rucos = torqueDot(r, u);
         real rusin1 = REAL_RSQRT(1 - rucos * rucos);
         dphidr = -torqueDot(trq, r);
         torqueCross(del, r, u);
         delsiz1 = REAL_RSQRT(torqueDot(del, del));
         torqueNormal(del, delsiz1);
         dphiddel = -torqueDot(trq, del);
         torqueCross(eps, del, u);
         for (int j = 0; j < 3; ++j) {
            real du = del[j] * dphidr * usiz1 * rusin1 + eps[j] * dphiddel * upcos * usiz1 * psiz1;
            deia[j] += du;
            deib[j] -= du;
            if CONSTEXPR (DO_V) {
               frcz[j] += du;
            }
         }

         r[0] = u[0] + w[0];
         r[1] = u[1] + w[1];
         r[2] = u[2] + w[2];
         rsiz1 = REAL_RSQRT(torqueDot(r, r));
         torqueNormal(r, rsiz1);
         real rvcos = torqueDot(r, v);
         real rvsin1 = REAL_RSQRT(1 - rvcos * rvcos);
         dphidr = -torqueDot(trq, r);
         torqueCross(del, r, v);
         delsiz1 = REAL_RSQRT(torqueDot(del, del));
         torqueNormal(del, delsiz1);
         dphiddel = -torqueDot(trq, del);
         torqueCross(eps, del, v);
         for (int j = 0; j < 3; ++j) {
            real dv = del[j] * dphidr * vsiz1 * rvsin1 + eps[j] * dphiddel * vpcos * vsiz1 * psiz1;
            deic[j] += dv;
            deib[j] -= dv;
            if CONSTEXPR (DO_V) {
               frcx[j] += dv;
            }
         }
      }

      atomic_add(deia[0], gx, ia);
      atomic_add(deia[1], gy, ia);
      atomic_add(deia[2], gz, ia);
      atomic_add(deib[0], gx, ib);
      atomic_add(deib[1], gy, ib);
      atomic_add(deib[2], gz, ib);
      if (axetyp != LFRM_Z_ONLY) {
         atomic_add(deic[0], gx, ic);
         atomic_add(deic[1], gy, ic);
         atomic_add(deic[2], gz, ic);
      }
      if (axetyp == LFRM_Z_BISECT || axetyp == LFRM_3_FOLD) {
         atomic_add(deid[0], gx, id);
         atomic_add(deid[1], gy, id);
         atomic_add(deid[2], gz, id);
      }

      if CONSTEXPR (DO_V) {
         const int iaz = (ia == -1) ? i : ia;
         const int iax = (ic == -1) ? i : ic;
         const int iay = (id == -1) ? i : id;

         real xiz = x[iaz] - x[i];
         real yiz = y[iaz] - y[i];
         real ziz = z[iaz] - z[i];

         real xix = x[iax] - x[i];
         real yix = y[iax] - y[i];
         real zix = z[iax] - z[i];

         real xiy = x[iay] - x[i];
         real yiy = y[iay] - y[i];
         real ziy = z[iay] - z[i];

         real vxx = xix * frcx[0] + xiy * frcy[0] + xiz * frcz[0];
         real vxy = 0.5f
            * (yix * frcx[0] + yiy * frcy[0] + yiz * frcz[0] + xix * frcx[1] + xiy * frcy[1] + xiz * frcz[1]);
         real vxz = 0.5f
            * (zix * frcx[0] + ziy * frcy[0] + ziz * frcz[0] + xix * frcx[2] + xiy * frcy[2] + xiz * frcz[2]);
         real vyy = yix * frcx[1] + yiy * frcy[1] + yiz * frcz[1];
         real vyz = 0.5f
            * (zix * frcx[1] + ziy * frcy[1] + ziz * frcz[1] + yix * frcx[2] + yiy * frcy[2] + yiz * frcz[2]);
         real vzz = zix * frcx[2] + ziy * frcy[2] + ziz * frcz[2];

         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_vir, ithread);
      } // end if CONSTEXPR (DO_V)
   }
}

void torque_cu(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz)
{
   if (vers & calc::virial) {
      auto ker = torque_cu1<true>;
      launch_k1b(g::s0, n, ker,  //
         n, vir_trq, gx, gy, gz, //
         x, y, z, zaxis, trqx, trqy, trqz);
   } else if (vers & calc::grad) {
      auto ker = torque_cu1<false>;
      launch_k1s(g::s0, n, ker,  //
         n, nullptr, gx, gy, gz, //
         x, y, z, zaxis, trqx, trqy, trqz);
   }
}
}
