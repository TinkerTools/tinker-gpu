#include "ff/energy.h"
#include "math/libfunc.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "seq/add.h"
#include "seq/settle.h"
#include <tinker/detail/units.hh>

namespace tinker {
static const int maxiter = 500;
static const double sor = 1.25;

template <class HTYPE>
static void constrain_acc(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold,
   const pos_prec* yold, const pos_prec* zold)
{
   if (nratmol <= 0)
      return;

   const double eps = rateps;

   #pragma acc parallel loop independent async\
           deviceptr(massinv,xnew,ynew,znew,vx,vy,vz,irat,krat,iratmol,\
                     xold,yold,zold)
   for (int im = 0; im < nratmol; ++im) {
      int mbegin = iratmol[im][0];
      int mend = iratmol[im][1];

      bool done = false;
      #pragma acc loop seq
      for (int niter = 0; niter < maxiter and not done; ++niter) {
         done = true;
         for (int i = mbegin; i < mend; ++i) {
            int ia = irat[i][0];
            int ib = irat[i][1];
            double xr, yr, zr, delta;
            xr = xnew[ib] - xnew[ia];
            yr = ynew[ib] - ynew[ia];
            zr = znew[ib] - znew[ia];
            delta = krat[i] * krat[i] - (xr * xr + yr * yr + zr * zr);
            if (fabs(delta) > eps) {
               done = false;
               double xo, yo, zo, dot, rma, rmb, term, xterm, yterm, zterm;
               xo = xold[ib] - xold[ia];
               yo = yold[ib] - yold[ia];
               zo = zold[ib] - zold[ia];
               dot = xr * xo + yr * yo + zr * zo;
               rma = massinv[ia];
               rmb = massinv[ib];
               term = 0.5 * sor * delta / ((rma + rmb) * dot);
               xterm = xo * term;
               yterm = yo * term;
               zterm = zo * term;
               xnew[ia] -= xterm * rma;
               ynew[ia] -= yterm * rma;
               znew[ia] -= zterm * rma;
               xnew[ib] += xterm * rmb;
               ynew[ib] += yterm * rmb;
               znew[ib] += zterm * rmb;
               if CONSTEXPR (not eq<HTYPE, SHAKE>()) {
                  rma /= dt;
                  rmb /= dt;
                  vx[ia] -= xterm * rma;
                  vy[ia] -= yterm * rma;
                  vz[ia] -= zterm * rma;
                  vx[ib] += xterm * rmb;
                  vy[ib] += yterm * rmb;
                  vz[ib] += zterm * rmb;
               }
            } // end if (delta > eps)
         }
      } // end (maxiter or done)
   }
}

template <class HTYPE>
static void settle_acc1(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold,
   const pos_prec* yold, const pos_prec* zold)
{
   if (nratwt <= 0)
      return;

   #pragma acc parallel loop independent async\
           deviceptr(xold,yold,zold,xnew,ynew,znew,vx,vy,vz,mass,\
                     iratwt,kratwt)
   for (int iw = 0; iw < nratwt; ++iw) {
      dk_settle1<HTYPE>(dt, iw, xold, yold, zold, xnew, ynew, znew, //
         vx, vy, vz, mass, iratwt, kratwt);
   }
}

template <class HTYPE>
static void constrainCH_acc1(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold,
   const pos_prec* yold, const pos_prec* zold)
{
   if (nratch <= 0)
      return;

   #pragma acc parallel loop independent vector_length(64) async\
           deviceptr(xold,yold,zold,xnew,ynew,znew,vx,vy,vz,massinv,\
                     iratch,kratch)
   for (int im = 0; im < nratch; ++im) {
      int ia, ib;
      double lab;
      double rma, rmb, invm;
      ia = iratch[im][0];
      ib = iratch[im][1];
      lab = kratch[im];
      rma = massinv[ia];
      rmb = massinv[ib];
      invm = rma + rmb;

      // vectors AB0, AB1
      // AB0 dot AB1
      double xb0, yb0, zb0, sqb0, xb1, yb1, zb1, sqb1, dot;
      xb0 = xold[ib] - xold[ia];
      yb0 = yold[ib] - yold[ia];
      zb0 = zold[ib] - zold[ia];
      sqb0 = xb0 * xb0 + yb0 * yb0 + zb0 * zb0;
      xb1 = xnew[ib] - xnew[ia];
      yb1 = ynew[ib] - ynew[ia];
      zb1 = znew[ib] - znew[ia];
      sqb1 = xb1 * xb1 + yb1 * yb1 + zb1 * zb1;
      dot = xb0 * xb1 + yb0 * yb1 + zb0 * zb1;

      // al lambda**2 - 2 * be lambda + ga = 0
      double al, be, ga, sqterm, lam;
      al = sqb0 * invm * invm;
      be = invm * dot;
      ga = sqb1 - lab * lab;
      sqterm = be * be - al * ga;
      lam = (be - sqrt(sqterm)) / al;

      // delta_A and delta_B
      double dxa, dya, dza, dxb, dyb, dzb;
      dxa = lam * xb0 * rma;
      dya = lam * yb0 * rma;
      dza = lam * zb0 * rma;
      dxb = -lam * xb0 * rmb;
      dyb = -lam * yb0 * rmb;
      dzb = -lam * zb0 * rmb;

      xnew[ia] += dxa;
      ynew[ia] += dya;
      znew[ia] += dza;
      xnew[ib] += dxb;
      ynew[ib] += dyb;
      znew[ib] += dzb;

      if CONSTEXPR (not eq<HTYPE, SHAKE>()) {
         double invdt = 1 / dt;
         vx[ia] += dxa * invdt;
         vy[ia] += dya * invdt;
         vz[ia] += dza * invdt;
         vx[ib] += dxb * invdt;
         vy[ib] += dyb * invdt;
         vz[ib] += dzb * invdt;
      }
   }
}

void rattle_acc(time_prec dt, const pos_prec* xold, const pos_prec* yold, const pos_prec* zold)
{
   constrain_acc<RATTLE>(dt, xpos, ypos, zpos, xold, yold, zold);
}

void rattleSettle_acc(time_prec dt, const pos_prec* xold, const pos_prec* yold, const pos_prec* zold)
{
   settle_acc1<RATTLE>(dt, xpos, ypos, zpos, xold, yold, zold);
}

void rattleCH_acc(time_prec dt, const pos_prec* xold, const pos_prec* yold, const pos_prec* zold)
{
   constrainCH_acc1<RATTLE>(dt, xpos, ypos, zpos, xold, yold, zold);
}

void shake_acc(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold, const pos_prec* yold,
   const pos_prec* zold)
{
   constrain_acc<SHAKE>(dt, xnew, ynew, znew, xold, yold, zold);
}

void shakeSettle_acc(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold,
   const pos_prec* yold, const pos_prec* zold)
{
   settle_acc1<SHAKE>(dt, xnew, ynew, znew, xold, yold, zold);
}

void shakeCH_acc(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold,
   const pos_prec* yold, const pos_prec* zold)
{
   constrainCH_acc1<SHAKE>(dt, xnew, ynew, znew, xold, yold, zold);
}

template <bool DO_V>
static void constrain2_acc(time_prec dt)
{
   if (nratmol <= 0)
      return;

   const double eps = rateps / dt;
   const double vterm = 2 / (dt * units::ekcal);
   size_t bufsize = bufferSize();

   #pragma acc parallel loop independent async\
           deviceptr(massinv,xpos,ypos,zpos,vx,vy,vz,vir_buf,irat,krat,iratmol)
   for (int im = 0; im < nratmol; ++im) {
      int mbegin = iratmol[im][0];
      int mend = iratmol[im][1];

      bool done = false;
      #pragma acc loop seq
      for (int niter = 0; niter < maxiter and not done; ++niter) {
         done = true;
         for (int i = mbegin; i < mend; ++i) {
            int ia = irat[i][0];
            int ib = irat[i][1];
            double xr, yr, zr, xv, yv, zv, dot, rma, rmb, term;
            xr = xpos[ib] - xpos[ia];
            yr = ypos[ib] - ypos[ia];
            zr = zpos[ib] - zpos[ia];
            xv = vx[ib] - vx[ia];
            yv = vy[ib] - vy[ia];
            zv = vz[ib] - vz[ia];
            dot = xr * xv + yr * yv + zr * zv;
            rma = massinv[ia];
            rmb = massinv[ib];
            term = -dot / ((rma + rmb) * krat[i] * krat[i]);
            if (fabs(term) > eps) {
               done = false;
               term *= sor;
               double xterm, yterm, zterm;
               xterm = xr * term;
               yterm = yr * term;
               zterm = zr * term;
               vx[ia] -= xterm * rma;
               vy[ia] -= yterm * rma;
               vz[ia] -= zterm * rma;
               vx[ib] += xterm * rmb;
               vy[ib] += yterm * rmb;
               vz[ib] += zterm * rmb;
               if CONSTEXPR (DO_V) {
                  size_t offset = im & (bufsize - 1);
                  double vxx = 0, vyx = 0, vzx = 0;
                  double vyy = 0, vzy = 0, vzz = 0;
                  xterm *= vterm;
                  yterm *= vterm;
                  zterm *= vterm;
                  vxx -= xr * xterm;
                  vyx -= yr * xterm;
                  vzx -= zr * xterm;
                  vyy -= yr * yterm;
                  vzy -= zr * yterm;
                  vzz -= zr * zterm;
                  atomic_add((real)vxx, (real)vyx, (real)vzx, (real)vyy, (real)vzy, (real)vzz, vir_buf, offset);
               }
            } // end if (delta > eps)
         }
      } // end (maxiter or done)
   }
}

template <bool DO_V>
static void settle2_acc1(time_prec dt)
{
   if (nratwt <= 0)
      return;

   size_t bufsize = bufferSize();

   #pragma acc parallel loop independent async\
           deviceptr(vx,vy,vz,xpos,ypos,zpos,mass,vir_buf,iratwt)
   for (int iw = 0; iw < nratwt; ++iw) {
      double vxx, vyx, vzx, vyy, vzy, vzz;
      dk_settle2<DO_V>(dt, iw, vx, vy, vz, xpos, ypos, zpos, mass, iratwt, //
         vxx, vyx, vzx, vyy, vzy, vzz);
      if CONSTEXPR (DO_V) {
         size_t offset = iw & (bufsize - 1);
         atomic_add((real)vxx, (real)vyx, (real)vzx, (real)vyy, (real)vzy, (real)vzz, vir_buf, offset);
      }
   }
}

template <bool DO_V>
void constrain2CH_acc1(time_prec dt)
{
   if (nratch <= 0)
      return;

   const double vterm = 2 / (dt * units::ekcal);
   size_t bufsize = bufferSize();

   #pragma acc parallel loop independent vector_length(64) async\
           deviceptr(massinv,xpos,ypos,zpos,vx,vy,vz,vir_buf,iratch)
   for (int im = 0; im < nratch; ++im) {
      int ia, ib;
      double rma, rmb;
      ia = iratch[im][0];
      ib = iratch[im][1];
      rma = massinv[ia];
      rmb = massinv[ib];

      // vectors AB3, vAB0, AB3 dot vAB0, -lam
      double xr, yr, zr, xv, yv, zv, dot, term;
      xr = xpos[ib] - xpos[ia];
      yr = ypos[ib] - ypos[ia];
      zr = zpos[ib] - zpos[ia];
      xv = vx[ib] - vx[ia];
      yv = vy[ib] - vy[ia];
      zv = vz[ib] - vz[ia];
      dot = xr * xv + yr * yv + zr * zv;
      double r2 = xr * xr + yr * yr + zr * zr;
      term = -dot / ((rma + rmb) * r2);

      double xterm, yterm, zterm;
      xterm = xr * term;
      yterm = yr * term;
      zterm = zr * term;
      vx[ia] -= xterm * rma;
      vy[ia] -= yterm * rma;
      vz[ia] -= zterm * rma;
      vx[ib] += xterm * rmb;
      vy[ib] += yterm * rmb;
      vz[ib] += zterm * rmb;
      if CONSTEXPR (DO_V) {
         size_t offset = im & (bufsize - 1);
         double vxx = 0, vyx = 0, vzx = 0;
         double vyy = 0, vzy = 0, vzz = 0;
         xterm *= vterm;
         yterm *= vterm;
         zterm *= vterm;
         vxx -= xr * xterm;
         vyx -= yr * xterm;
         vzx -= zr * xterm;
         vyy -= yr * yterm;
         vzy -= zr * yterm;
         vzz -= zr * zterm;
         atomic_add((real)vxx, (real)vyx, (real)vzx, (real)vyy, (real)vzy, (real)vzz, vir_buf, offset);
      }
   }
}

void rattle2_acc(time_prec dt, bool do_v)
{
   if (do_v) {
      constrain2_acc<true>(dt);
   } else {
      constrain2_acc<false>(dt);
   }
}

void rattle2Settle_acc(time_prec dt, bool do_v)
{
   if (do_v)
      settle2_acc1<true>(dt);
   else
      settle2_acc1<false>(dt);
}

void rattle2CH_acc(time_prec dt, bool do_v)
{
   if (do_v) {
      constrain2CH_acc1<true>(dt);
   } else {
      constrain2CH_acc1<false>(dt);
   }
}
}

namespace tinker {
void hcVirial_acc()
{
   int nmol = rattle_dmol.nmol;
   auto* molmass = rattle_dmol.molmass;
   auto* imol = rattle_dmol.imol;
   auto* kmol = rattle_dmol.kmol;

   double mvxx = 0, mvyy = 0, mvzz = 0, mvxy = 0, mvxz = 0, mvyz = 0;
   #pragma acc parallel loop independent async\
               copy(mvxx,mvyy,mvzz,mvxy,mvxz,mvyz)\
               reduction(+:mvxx,mvyy,mvzz,mvxy,mvxz,mvyz)\
               deviceptr(molmass,imol,kmol,mass,xpos,ypos,zpos,gx,gy,gz)
   for (int im = 0; im < nmol; ++im) {
      double vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
      double igx, igy, igz;             // atomic gradients
      pos_prec irx, iry, irz;           // atomic positions
      double mgx = 0, mgy = 0, mgz = 0; // molecular gradients
      pos_prec rx = 0, ry = 0, rz = 0;  // molecular positions
      int start = imol[im][0];
      int end = imol[im][1];
      #pragma acc loop seq
      for (int i = start; i < end; ++i) {
         int k = kmol[i];
         igx = toFloatGrad<double>(gx[k]);
         igy = toFloatGrad<double>(gy[k]);
         igz = toFloatGrad<double>(gz[k]);
         irx = xpos[k];
         iry = ypos[k];
         irz = zpos[k];
         vxx -= igx * irx;
         vyy -= igy * iry;
         vzz -= igz * irz;
         vxy -= 0.5 * (igx * iry + igy * irx);
         vxz -= 0.5 * (igx * irz + igz * irx);
         vyz -= 0.5 * (igy * irz + igz * iry);

         mgx += igx;
         mgy += igy;
         mgz += igz;
         auto massk = mass[k];
         rx += massk * irx;
         ry += massk * iry;
         rz += massk * irz;
      }
      auto mmassinv = 1 / molmass[im];
      vxx += mgx * rx * mmassinv;
      vyy += mgy * ry * mmassinv;
      vzz += mgz * rz * mmassinv;
      vxy += 0.5 * (mgx * ry + mgy * rx) * mmassinv;
      vxz += 0.5 * (mgx * rz + mgz * rx) * mmassinv;
      vyz += 0.5 * (mgy * rz + mgz * ry) * mmassinv;
      mvxx += vxx;
      mvyy += vyy;
      mvzz += vzz;
      mvxy += vxy;
      mvxz += vxz;
      mvyz += vyz;
   }
   #pragma acc wait

   hc_vir[0] = mvxx + vir[0];
   hc_vir[1] = mvxy + vir[1];
   hc_vir[2] = mvxz + vir[2];
   hc_vir[3] = mvxy + vir[3];
   hc_vir[4] = mvyy + vir[4];
   hc_vir[5] = mvyz + vir[5];
   hc_vir[6] = mvxz + vir[6];
   hc_vir[7] = mvyz + vir[7];
   hc_vir[8] = mvzz + vir[8];
}

void hcCenterOfMass_acc(const pos_prec* ax, const pos_prec* ay, const pos_prec* az, pos_prec* mx, pos_prec* my,
   pos_prec* mz)
{
   const int nmol = rattle_dmol.nmol;
   const auto* imol = rattle_dmol.imol;
   const auto* kmol = rattle_dmol.kmol;
   const auto* mfrac = ratcom_massfrac;
   #pragma acc parallel loop independent async\
               deviceptr(ax,ay,az,mx,my,mz,mfrac,imol,kmol)
   for (int im = 0; im < nmol; ++im) {
      int start = imol[im][0];
      int end = imol[im][1];
      pos_prec tx = 0, ty = 0, tz = 0;
      #pragma acc loop seq
      for (int i = start; i < end; ++i) {
         int k = kmol[i];
         auto frk = mfrac[k];
         tx += frk * ax[k];
         ty += frk * ay[k];
         tz += frk * az[k];
      }
      mx[im] = tx;
      my[im] = ty;
      mz[im] = tz;
   }
}

void hcVelIso_acc(vel_prec scal)
{
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz,ratcom_vx,ratcom_vy,ratcom_vz,molec)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      vx[i] += scal * ratcom_vx[im];
      vy[i] += scal * ratcom_vy[im];
      vz[i] += scal * ratcom_vz[im];
   }
}

void hcVelAn_acc(vel_prec scal[3][3])
{
   auto s00 = scal[0][0], s01 = scal[0][1], s02 = scal[0][2];
   auto s10 = scal[1][0], s11 = scal[1][1], s12 = scal[1][2];
   auto s20 = scal[2][0], s21 = scal[2][1], s22 = scal[2][2];
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz,ratcom_vx,ratcom_vy,ratcom_vz,molec)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      auto xm = ratcom_vx[im], ym = ratcom_vy[im], zm = ratcom_vz[im];
      vx[i] += s00 * xm + s01 * ym + s02 * zm;
      vy[i] += s10 * xm + s11 * ym + s12 * zm;
      vz[i] += s20 * xm + s21 * ym + s22 * zm;
   }
}

void hcPosIso_acc(pos_prec s)
{
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
           deviceptr(xpos,ypos,zpos,ratcom_x,ratcom_y,ratcom_z,molec)
   for (int i = 0; i < n; ++i) {
      auto k = molec[i];
      xpos[i] += ratcom_x[k] * s;
      ypos[i] += ratcom_y[k] * s;
      zpos[i] += ratcom_z[k] * s;
   }
}

void hcPosAn_acc(pos_prec (*scal)[3])
{
   double s00, s01, s02;
   double s10, s11, s12;
   double s20, s21, s22;
   s00 = scal[0][0], s01 = scal[0][1], s02 = scal[0][2];
   s10 = scal[1][0], s11 = scal[1][1], s12 = scal[1][2];
   s20 = scal[2][0], s21 = scal[2][1], s22 = scal[2][2];
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
           deviceptr(xpos,ypos,zpos,ratcom_x,ratcom_y,ratcom_z,molec)
   for (int i = 0; i < n; ++i) {
      auto k = molec[i];
      auto xc = ratcom_x[k];
      auto yc = ratcom_y[k];
      auto zc = ratcom_z[k];
      auto xd = s00 * xc + s01 * yc + s02 * zc;
      auto yd = s10 * xc + s11 * yc + s12 * zc;
      auto zd = s20 * xc + s21 * yc + s22 * zc;
      xpos[i] += xd;
      ypos[i] += yd;
      zpos[i] += zd;
   }
}
}
