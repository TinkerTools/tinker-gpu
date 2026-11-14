#pragma once
#include "ff/colvar.h"
#include "seq/add.h"
#include "seq/geom.h"

namespace tinker {
/// \ingroup ff
/// The share of entry \c j of #Colvar::satom in the center of its set \c s:
/// its mass-weighted offset from the first atom of the set (#geomGroupOffset),
/// zero for the first atom itself.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSetTerm(int& restrict s,
                          real& restrict dx,
                          real& restrict dy,
                          real& restrict dz,
                          int j,
                          const int* restrict sown,
                          const int (*restrict srange)[2],
                          const int* restrict satom,
                          const real* restrict sminv,
                          const real* restrict x,
                          const real* restrict y,
                          const real* restrict z,
                          const double* restrict mass,
                          const int* restrict molec,
                          TINKER_IMAGE_PARAMS)
{
   s = sown[j];
   const int j1 = srange[s][0];
   dx = 0, dy = 0, dz = 0;
   if (j != j1) {
      const int k = satom[j];
      geomGroupOffset(dx, dy, dz, k, satom[j1], (real)mass[k] * sminv[s], x, y, z, molec, TINKER_IMAGE_ARGS);
   }
}

/// \ingroup ff
/// Center of set \c s from the summed offsets of its atoms.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSetCenter(real& restrict cx,
                            real& restrict cy,
                            real& restrict cz,
                            int s,
                            const grad_prec (*restrict ssum)[3],
                            const int (*restrict srange)[2],
                            const int* restrict satom,
                            const real* restrict x,
                            const real* restrict y,
                            const real* restrict z)
{
   const int k1 = satom[srange[s][0]];
   cx = x[k1] + toFloatGrad<real>(ssum[s][0]);
   cy = y[k1] + toFloatGrad<real>(ssum[s][1]);
   cz = z[k1] + toFloatGrad<real>(ssum[s][2]);
}

/// \ingroup ff
/// DISTANCE, slot \c g of CV \c i: the distance between the centers of sets A
/// and B, imaged if their first atoms are in different molecules. The slots
/// are the atoms of A, then the atoms of B; a zero distance has a zero
/// gradient. The first slot also writes q and W. Returns q and dq/dx of the
/// slot in \c q and \c gs.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarDistanceSlot(real& restrict q,
                               real& restrict gsx,
                               real& restrict gsy,
                               real& restrict gsz,
                               int g,
                               int i,
                               real* restrict val,
                               real (*restrict w)[9],
                               real (*restrict grad)[3],
                               const grad_prec (*restrict ssum)[3],
                               const int (*restrict iarg)[4],
                               const int (*restrict slot)[2],
                               const int (*restrict srange)[2],
                               const int* restrict satom,
                               const real* restrict sminv,
                               const real* restrict x,
                               const real* restrict y,
                               const real* restrict z,
                               const double* restrict mass,
                               const int* restrict molec,
                               TINKER_IMAGE_PARAMS)
{
   const int sa = iarg[i][0], sb = iarg[i][1];
   real xa, ya, za, xb, yb, zb;
   colvarSetCenter(xa, ya, za, sa, ssum, srange, satom, x, y, z);
   colvarSetCenter(xb, yb, zb, sb, ssum, srange, satom, x, y, z);
   real xr = xa - xb, yr = ya - yb, zr = za - zb;
   const int ja1 = srange[sa][0], ja2 = srange[sa][1], jb1 = srange[sb][0];
   if (molec[satom[ja1]] != molec[satom[jb1]])
      image(xr, yr, zr);
   const real r = REAL_SQRT(xr * xr + yr * yr + zr * zr);
   const real rinv = (r == 0 ? 0 : REAL_RECIP(r));
   const real ux = xr * rinv, uy = yr * rinv, uz = zr * rinv;

   const int g0 = slot[i][0];
   if (g == g0) {
      val[i] = r;
      w[i][0] = xr * ux, w[i][1] = xr * uy, w[i][2] = xr * uz;
      w[i][3] = yr * ux, w[i][4] = yr * uy, w[i][5] = yr * uz;
      w[i][6] = zr * ux, w[i][7] = zr * uy, w[i][8] = zr * uz;
   }

   const int local = g - g0;
   const int na = ja2 - ja1;
   const int j = (local < na ? ja1 + local : jb1 + local - na);
   const real wt = (real)mass[satom[j]] * (local < na ? sminv[sa] : -sminv[sb]);
   q = r;
   gsx = wt * ux;
   gsy = wt * uy;
   gsz = wt * uz;
   grad[g][0] = gsx;
   grad[g][1] = gsy;
   grad[g][2] = gsz;
}

/// \ingroup ff
/// Gradient slot \c g, by the type of its CV; returns q of the CV and dq/dx of
/// the slot.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSlot1(real& restrict q,
                        real& restrict gsx,
                        real& restrict gsy,
                        real& restrict gsz,
                        int g,
                        const int* restrict gcv,
                        const int* restrict type,
                        real* restrict val,
                        real (*restrict w)[9],
                        real (*restrict grad)[3],
                        const grad_prec (*restrict ssum)[3],
                        const int (*restrict iarg)[4],
                        const int (*restrict slot)[2],
                        const int (*restrict srange)[2],
                        const int* restrict satom,
                        const real* restrict sminv,
                        const real* restrict x,
                        const real* restrict y,
                        const real* restrict z,
                        const double* restrict mass,
                        const int* restrict molec,
                        TINKER_IMAGE_PARAMS)
{
   const int i = gcv[g];
   q = 0, gsx = 0, gsy = 0, gsz = 0;
   if (type[i] == (int)ColvarType::DISTANCE)
      colvarDistanceSlot(q, gsx, gsy, gsz, g, i, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass, molec, TINKER_IMAGE_ARGS);
}

/// \ingroup ff
/// The LAMBDA-CV restraint E = K (q - q0)^2 of the CV of slot \c g, with
/// q0 = qmin + g(lambda) dq and g, g', g'' in \c lg, \c lg1, \c lg2: the slot adds
/// the gradient of its atom and the derivative of dE/dL over it; the first
/// slot of the CV adds the energy, the virial from W, dE/dL, d2E/dL2 and the
/// lambda derivative of the virial.
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void colvarCouple(int g,
                  size_t offset,
                  real q,
                  real gsx,
                  real gsy,
                  real gsz,
                  const int* restrict gcv,
                  const int (*restrict slot)[2],
                  const int* restrict gatom,
                  const real (*restrict couple)[3],
                  const real (*restrict w)[9],
                  real lg,
                  real lg1,
                  real lg2,
                  EnergyBuffer restrict ecv,
                  VirialBuffer restrict vir_ecv,
                  grad_prec* restrict decvx,
                  grad_prec* restrict decvy,
                  grad_prec* restrict decvz,
                  EnergyBuffer restrict dl1,
                  EnergyBuffer restrict dl2,
                  VirialBuffer restrict dvirdl,
                  grad_prec* restrict dfdlx,
                  grad_prec* restrict dfdly,
                  grad_prec* restrict dfdlz)
{
   const int i = gcv[g];
   const real k = couple[i][0];
   const real dq = couple[i][2];
   const real dev = q - (couple[i][1] + lg * dq); // q - q0
   const real dq0 = lg1 * dq;                     // dq0/dL
   const real de = 2 * k * dev;                   // dE/dq
   const real fdl = -2 * k * dq0;                 // d(dE/dL)/dq

   if CONSTEXPR (Ver::g) {
      const int ka = gatom[g];
      atomic_add(de * gsx, decvx, ka);
      atomic_add(de * gsy, decvy, ka);
      atomic_add(de * gsz, decvz, ka);
      if CONSTEXPR (Ver::g_dlmda) {
         atomic_add(fdl * gsx, dfdlx, ka);
         atomic_add(fdl * gsy, dfdly, ka);
         atomic_add(fdl * gsz, dfdlz, ka);
      }
   }

   if (g != slot[i][0])
      return;
   if CONSTEXPR (Ver::e)
      atomic_add(k * dev * dev, ecv, offset);
   if CONSTEXPR (Ver::v)
      atomic_add(de * w[i][0], de * w[i][3], de * w[i][6], de * w[i][4], de * w[i][7], de * w[i][8], vir_ecv, offset);
   if CONSTEXPR (Ver::e_dlmda1)
      atomic_add(fdl * dev, dl1, offset);
   if CONSTEXPR (Ver::e_dlmda2)
      atomic_add(2 * k * dq0 * dq0 - de * lg2 * dq, dl2, offset);
   if CONSTEXPR (Ver::v_dlmda)
      atomic_add(fdl * w[i][0], fdl * w[i][3], fdl * w[i][6], fdl * w[i][4], fdl * w[i][7], fdl * w[i][8], dvirdl, offset);
}
}
