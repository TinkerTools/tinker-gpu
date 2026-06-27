#include "ff/atom.h"
#include "ff/evalence.h"
#include "ff/hippo/cflux.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "tool/darray.h"

namespace tinker {
static void bndchg_acc1()
{
   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,ibnd,bl,bflx,pdelta)
   for (int i = 0; i < nbond; ++i) {
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
   } // end for (int i)
}

static void angchg_acc1()
{
   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,iang,anat,aflx,abflx,bl,balist,pdelta)
   for (int i = 0; i < nangle; ++i) {
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

      real dot, angle, cosine;
      if (rab != 0 and rcb != 0) {
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
   } // end for (int i)
}

void alterchg_acc()
{
   darray::zero(g::q0, n, pdelta);

   bndchg_acc1();
   angchg_acc1();

   // alter monopoles and charge penetration
    #pragma acc parallel loop independent async\
           deviceptr(pval,pval0,pdelta,pole,mono0)
   for (int i = 0; i < n; ++i) {
      pval[i] = pval0[i] + pdelta[i];
      pole[i][0] = mono0[i] + pdelta[i];
   }
}
}

namespace tinker {
static void dcfluxBnd_acc()
{
   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,ibnd,bflx,pot,\
               decfx,decfy,decfz)
   for (int i = 0; i < nbond; ++i) {
      int ia = ibnd[i][0];
      int ib = ibnd[i][1];

      real pb = bflx[i];
      real xab = x[ia] - x[ib];
      real yab = y[ia] - y[ib];
      real zab = z[ia] - z[ib];
      real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
      real dpot = pot[ia] - pot[ib];
      pb = pb / rab;

      real fx = dpot * pb * xab;
      real fy = dpot * pb * yab;
      real fz = dpot * pb * zab;
      atomic_add(-fx, decfx, ia);
      atomic_add(-fy, decfy, ia);
      atomic_add(-fz, decfz, ia);
      atomic_add(fx, decfx, ib);
      atomic_add(fy, decfy, ib);
      atomic_add(fz, decfz, ib);
   }
}

static void dcfluxAng_acc()
{
   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,iang,aflx,abflx,pot,decfx,decfy,decfz)
   for (int i = 0; i < nangle; ++i) {
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
      real dba[3];
      dba[0] = xab;
      dba[1] = yab;
      dba[2] = zab;

      real rbc2 = xcb * xcb + ycb * ycb + zcb * zcb;
      real rbc = REAL_SQRT(rbc2);
      real rbc3 = rbc2 * rbc;
      real dbc[3];
      dbc[0] = xcb;
      dbc[1] = ycb;
      dbc[2] = zcb;

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

      real dot = dba[0] * dbc[0] + dba[1] * dbc[1] + dba[2] * dbc[2];
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
      atomic_add((fax + fax2), decfx, ia);
      atomic_add((fay + fay2), decfy, ia);
      atomic_add((faz + faz2), decfz, ia);
      atomic_add((fbx + fbx2), decfx, ib);
      atomic_add((fby + fby2), decfy, ib);
      atomic_add((fbz + fbz2), decfz, ib);
      atomic_add((fcx + fcx2), decfx, ic);
      atomic_add((fcy + fcy2), decfy, ic);
      atomic_add((fcz + fcz2), decfz, ic);
   }
}

template <int DO_V>
static void dcflux_acc1(grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   VirialBuffer restrict vir)
{
   auto bufsize = bufferSize();

   #pragma acc parallel loop independent async\
               deviceptr(decfx,decfy,decfz)
   for (int i = 0; i < n; ++i) {
      decfx[i] = 0;
      decfy[i] = 0;
      decfz[i] = 0;
   }

   dcfluxBnd_acc();
   dcfluxAng_acc();

   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,decfx,decfy,decfz,\
               gx,gy,gz,vir)
   for (int i = 0; i < n; ++i) {
      atomic_add(decfx[i], gx, i);
      atomic_add(decfy[i], gy, i);
      atomic_add(decfz[i], gz, i);

      if CONSTEXPR (DO_V) {
         real vxx = x[i] * decfx[i];
         real vyx = y[i] * decfx[i];
         real vzx = z[i] * decfx[i];
         real vyy = y[i] * decfy[i];
         real vzy = z[i] * decfy[i];
         real vzz = z[i] * decfz[i];
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir, i & (bufsize - 1));
      }
   }
}

void dcflux_acc(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer vir)
{
   if (vers & calc::virial)
      dcflux_acc1<1>(gx, gy, gz, vir);
   else
      dcflux_acc1<0>(gx, gy, gz, nullptr);
}
}
