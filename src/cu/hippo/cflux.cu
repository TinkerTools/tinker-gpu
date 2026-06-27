#include "ff/evalence.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "math/zero.h"
#include "seq/add.h"
#include "seq/launch.h"

namespace tinker {
__global__
static void bndangChg_cu1(real* restrict pdelta,                                                 //
   const real* restrict x, const real* restrict y, const real* restrict z,                       //
   int nbond, const int (*restrict ibnd)[2], const real* restrict bl, const real* restrict bflx, //
   int nangle, const int (*restrict iang)[4], const real* restrict anat,                         //
   const real (*restrict aflx)[2], const real (*restrict abflx)[2], const int (*restrict balist)[2])
{
   for (int i = ITHREAD; i < nbond; i += STRIDE) {
      int ia = ibnd[i][0];
      int ib = ibnd[i][1];

      real pb = bflx[i];
      real ideal = bl[i];
      real xab = x[ia] - x[ib];
      real yab = y[ia] - y[ib];
      real zab = z[ia] - z[ib];

      real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
      real dq = pb * (rab - ideal);

      atomic_add(-dq, pdelta, ia);
      atomic_add(dq, pdelta, ib);
   }

   for (int i = ITHREAD; i < nangle; i += STRIDE) {
      int ia = iang[i][0];
      int ib = iang[i][1];
      int ic = iang[i][2];

      real pa1 = aflx[i][0];
      real pa2 = aflx[i][1];
      real pb1 = abflx[i][0];
      real pb2 = abflx[i][1];
      real ideal = anat[i];
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
      real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
      real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);

      if (rab != 0 and rcb != 0) {
         real dot, angle, cosine;
         dot = xab * xcb + yab * ycb + zab * zcb;
         cosine = dot / (rab * rcb);
         cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
         angle = radian * REAL_ACOS(cosine);

         int ab = balist[i][0];
         int cb = balist[i][1];
         real rab0 = bl[ab];
         real rcb0 = bl[cb];
         real dq1 = pb1 * (rcb - rcb0) + pa1 * (angle - ideal) / radian;
         real dq2 = pb2 * (rab - rab0) + pa2 * (angle - ideal) / radian;
         atomic_add(dq1, pdelta, ia);
         atomic_add(dq2, pdelta, ic);
         atomic_add(-(dq1 + dq2), pdelta, ib);
      }
   }
}

__global__
static void bndangChg_cu2(int n, real* restrict pval, real (*restrict pole)[MPL_TOTAL], //
   const real* restrict pval0, const real* restrict mono0, const real* restrict pdelta)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      pval[i] = pval0[i] + pdelta[i];
      pole[i][0] = mono0[i] + pdelta[i];
   }
}

void alterchg_cu()
{
   darray::zero(g::q0, n, pdelta);

   auto nparall = std::max(nbond, nangle);
   if (nparall > 0) {
      launch_k1s(g::s0, nparall, bndangChg_cu1, //
         pdelta, x, y, z,                       //
         nbond, ibnd, bl, bflx,                 //
         nangle, iang, anat, aflx, abflx, balist);
   }

   launch_k1s(g::s0, n, bndangChg_cu2, //
      n, pval, pole, pval0, mono0, pdelta);
}
}

namespace tinker {
template <int DO_V>
__global__
static void dcfluxBndAng_cu1(VirialBuffer restrict vir, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, const real* restrict x, const real* restrict y, const real* restrict z,
   const real* restrict pot,                                            //
   int nbond, const int (*restrict ibnd)[2], const real* restrict bflx, //
   int nangle, const int (*restrict iang)[4], const real (*restrict aflx)[2], const real (*restrict abflx)[2])
{
   int ithread = ITHREAD;
   real vxx = 0, vyx = 0, vzx = 0, vyy = 0, vzy = 0, vzz = 0;

   for (int i = ithread; i < nbond; i += STRIDE) {
      int ia = ibnd[i][0];
      int ib = ibnd[i][1];

      real xab = x[ia] - x[ib];
      real yab = y[ia] - y[ib];
      real zab = z[ia] - z[ib];
      real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
      real dpot = pot[ia] - pot[ib];
      real pb = bflx[i] / rab;

      real fx = dpot * pb * xab;
      real fy = dpot * pb * yab;
      real fz = dpot * pb * zab;
      atomic_add(-fx, gx, ia);
      atomic_add(-fy, gy, ia);
      atomic_add(-fz, gz, ia);
      atomic_add(fx, gx, ib);
      atomic_add(fy, gy, ib);
      atomic_add(fz, gz, ib);

      if CONSTEXPR (DO_V) {
         vxx += -xab * fx;
         vyx += -yab * fx;
         vzx += -zab * fx;
         vyy += -yab * fy;
         vzy += -zab * fy;
         vzz += -zab * fz;
      }
   }

   for (int i = ithread; i < nangle; i += STRIDE) {
      int ia = iang[i][0];
      int ib = iang[i][1];
      int ic = iang[i][2];

      real pa1 = aflx[i][0];
      real pa2 = aflx[i][1];
      real pb1 = abflx[i][0];
      real pb2 = abflx[i][1];
      real xab = x[ia] - x[ib];
      real yab = y[ia] - y[ib];
      real zab = z[ia] - z[ib];
      real xcb = x[ic] - x[ib];
      real ycb = y[ic] - y[ib];
      real zcb = z[ic] - z[ib];

      real rba2 = xab * xab + yab * yab + zab * zab;
      real rba = REAL_SQRT(rba2);
      real rba3 = rba2 * rba;

      real rbc2 = xcb * xcb + ycb * ycb + zcb * zcb;
      real rbc = REAL_SQRT(rbc2);
      real rbc3 = rbc2 * rbc;

      real dpota = pot[ia] - pot[ib];
      real dpotc = pot[ic] - pot[ib];
      pb1 *= dpota;
      pb2 *= dpotc;

      real c1 = pb2 / rba;
      real c2 = pb1 / rbc;
      real fax = c1 * xab;
      real fay = c1 * yab;
      real faz = c1 * zab;
      real fcx = c2 * xcb;
      real fcy = c2 * ycb;
      real fcz = c2 * zcb;
      real fbx = -(fax + fcx);
      real fby = -(fay + fcy);
      real fbz = -(faz + fcz);

      real dot = xab * xcb + yab * ycb + zab * zcb;
      real term = -rba * rbc / REAL_SQRT(rba2 * rbc2 - dot * dot);
      real fterm = term * (dpota * pa1 + dpotc * pa2);
      c1 = 1 / (rba * rbc);
      c2 = dot / (rba3 * rbc);
      real c3 = dot / (rbc3 * rba);
      real fax2 = fterm * (c1 * xcb - c2 * xab);
      real fay2 = fterm * (c1 * ycb - c2 * yab);
      real faz2 = fterm * (c1 * zcb - c2 * zab);
      real fcx2 = fterm * (c1 * xab - c3 * xcb);
      real fcy2 = fterm * (c1 * yab - c3 * ycb);
      real fcz2 = fterm * (c1 * zab - c3 * zcb);
      real fbx2 = -(fax2 + fcx2);
      real fby2 = -(fay2 + fcy2);
      real fbz2 = -(faz2 + fcz2);
      atomic_add((fax + fax2), gx, ia);
      atomic_add((fay + fay2), gy, ia);
      atomic_add((faz + faz2), gz, ia);
      atomic_add((fbx + fbx2), gx, ib);
      atomic_add((fby + fby2), gy, ib);
      atomic_add((fbz + fbz2), gz, ib);
      atomic_add((fcx + fcx2), gx, ic);
      atomic_add((fcy + fcy2), gy, ic);
      atomic_add((fcz + fcz2), gz, ic);

      if CONSTEXPR (DO_V) {
         vxx += x[ia] * (fax + fax2) + x[ib] * (fbx + fbx2) + x[ic] * (fcx + fcx2);
         vyx += y[ia] * (fax + fax2) + y[ib] * (fbx + fbx2) + y[ic] * (fcx + fcx2);
         vzx += z[ia] * (fax + fax2) + z[ib] * (fbx + fbx2) + z[ic] * (fcx + fcx2);
         vyy += y[ia] * (fay + fay2) + y[ib] * (fby + fby2) + y[ic] * (fcy + fcy2);
         vzy += z[ia] * (fay + fay2) + z[ib] * (fby + fby2) + z[ic] * (fcy + fcy2);
         vzz += z[ia] * (faz + faz2) + z[ib] * (fbz + fbz2) + z[ic] * (fcz + fcz2);
      }
   }

   if CONSTEXPR (DO_V) {
      atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir, ithread);
   }
}

template <int DO_V>
static void dcflux_cu1(grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   VirialBuffer restrict vir)
{
   auto nparall = std::max(nbond, nangle);
   if (nparall > 0) {
      launch_k1b(g::s0, nparall, dcfluxBndAng_cu1<DO_V>, //
         vir, gx, gy, gz, x, y, z, pot,                  //
         nbond, ibnd, bflx,                              //
         nangle, iang, aflx, abflx);
   }
}

void dcflux_cu(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer vir)
{
   if (vers & calc::virial)
      dcflux_cu1<1>(gx, gy, gz, vir);
   else
      dcflux_cu1<0>(gx, gy, gz, nullptr);
}
}
