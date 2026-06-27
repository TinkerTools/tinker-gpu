#include "ff/atom.h"
#include "ff/box.h"
#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "seq/add.h"
#include "seq/bsplgen.h"

namespace tinker {
template <class T>
static void gridPut_acc(PMEUnit pme_u, real* ptr1, real* ptr2)
{
   auto& st = *pme_u;
   auto* qgrid = st.qgrid;

   MAYBE_UNUSED const real* pchg = ptr1;
   MAYBE_UNUSED const real(*fmp)[10] = (real(*)[10])ptr1;
   MAYBE_UNUSED const real(*fuind)[3] = (real(*)[3])ptr1;
   MAYBE_UNUSED const real(*fuinp)[3] = (real(*)[3])ptr2;

   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const int bsorder = st.bsorder;
   assert(bsorder <= 5);

   darray::zero(g::q0, 2 * nfft1 * nfft2 * nfft3, st.qgrid);

   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(pchg,fmp,fuind,fuinp,x,y,z,qgrid)
   for (int i = 0; i < n; ++i) {
      real thetai1[4 * 5];
      real thetai2[4 * 5];
      real thetai3[4 * 5];

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];

      // map fractional coordinate w from [-0.5 + k, 0.5 + k) to [0, 1)
      // w -> (w + 0.5) - FLOOR(w + 0.5)
      // see also subroutine bspline_fill in pmestuf.f

      real w1 = xi * recipa.x + yi * recipa.y + zi * recipa.z;
      w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
      real fr1 = nfft1 * w1;
      int igrid1 = REAL_FLOOR(fr1);
      w1 = fr1 - igrid1;

      real w2 = xi * recipb.x + yi * recipb.y + zi * recipb.z;
      w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
      real fr2 = nfft2 * w2;
      int igrid2 = REAL_FLOOR(fr2);
      w2 = fr2 - igrid2;

      real w3 = xi * recipc.x + yi * recipc.y + zi * recipc.z;
      w3 = w3 + 0.5f - REAL_FLOOR(w3 + 0.5f);
      real fr3 = nfft3 * w3;
      int igrid3 = REAL_FLOOR(fr3);
      w3 = fr3 - igrid3;

      igrid1 = igrid1 - bsorder + 1;
      igrid2 = igrid2 - bsorder + 1;
      igrid3 = igrid3 - bsorder + 1;
      igrid1 += (igrid1 < 0 ? nfft1 : 0);
      igrid2 += (igrid2 < 0 ? nfft2 : 0);
      igrid3 += (igrid3 < 0 ? nfft3 : 0);

      if CONSTEXPR (eq<T, PCHG>() || eq<T, DISP>()) {
         real pchgi = pchg[i];
         if (pchgi == 0)
            continue;

         bsplgen<1>(w1, thetai1, bsorder);
         bsplgen<1>(w2, thetai2, bsorder);
         bsplgen<1>(w3, thetai3, bsorder);

         #pragma acc loop seq
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz] * pchgi;
            #pragma acc loop seq
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy] * v0;
               #pragma acc loop seq
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real term = thetai1[4 * ix] * u0;
                  atomic_add(term, qgrid, 2 * index);
               }
            } // end for (int iy)
         }
      } // end if (PCHG or DISP)

      if CONSTEXPR (eq<T, MPOLE>()) {
         bsplgen<3>(w1, thetai1, bsorder);
         bsplgen<3>(w2, thetai2, bsorder);
         bsplgen<3>(w3, thetai3, bsorder);

         real fmpi0 = fmp[i][MPL_PME_0];
         real fmpix = fmp[i][MPL_PME_X];
         real fmpiy = fmp[i][MPL_PME_Y];
         real fmpiz = fmp[i][MPL_PME_Z];
         real fmpixx = fmp[i][MPL_PME_XX];
         real fmpiyy = fmp[i][MPL_PME_YY];
         real fmpizz = fmp[i][MPL_PME_ZZ];
         real fmpixy = fmp[i][MPL_PME_XY];
         real fmpixz = fmp[i][MPL_PME_XZ];
         real fmpiyz = fmp[i][MPL_PME_YZ];
         #pragma acc loop seq
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            real v2 = thetai3[4 * iz + 2];
            #pragma acc loop seq
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy];
               real u1 = thetai2[4 * iy + 1];
               real u2 = thetai2[4 * iy + 2];
               // fmp: 0, x, y, z, xx, yy, zz, xy, xz, yz
               //      1, 2, 3, 4,  5,  6,  7,  8,  9, 10
               real term0 = fmpi0 * u0 * v0 + fmpiy * u1 * v0 + fmpiz * u0 * v1 + fmpiyy * u2 * v0 + fmpizz * u0 * v2
                  + fmpiyz * u1 * v1;
               real term1 = fmpix * u0 * v0 + fmpixy * u1 * v0 + fmpixz * u0 * v1;
               real term2 = fmpixx * u0 * v0;
               #pragma acc loop seq
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real t0 = thetai1[4 * ix];
                  real t1 = thetai1[4 * ix + 1];
                  real t2 = thetai1[4 * ix + 2];
                  atomic_add(term0 * t0 + term1 * t1 + term2 * t2, qgrid, 2 * index);
               }
            } // end for (int iy)
         }
      } // end if (MPOLE)

      if CONSTEXPR (eq<T, UIND>()) {
         bsplgen<2>(w1, thetai1, bsorder);
         bsplgen<2>(w2, thetai2, bsorder);
         bsplgen<2>(w3, thetai3, bsorder);

         real fuindi0 = fuind[i][0];
         real fuindi1 = fuind[i][1];
         real fuindi2 = fuind[i][2];
         real fuinpi0 = fuinp[i][0];
         real fuinpi1 = fuinp[i][1];
         real fuinpi2 = fuinp[i][2];
         #pragma acc loop seq
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            #pragma acc loop seq
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy];
               real u1 = thetai2[4 * iy + 1];
               real term01 = fuindi1 * u1 * v0 + fuindi2 * u0 * v1;
               real term11 = fuindi0 * u0 * v0;
               real term02 = fuinpi1 * u1 * v0 + fuinpi2 * u0 * v1;
               real term12 = fuinpi0 * u0 * v0;
               #pragma acc loop seq
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real t0 = thetai1[4 * ix];
                  real t1 = thetai1[4 * ix + 1];
                  atomic_add(term01 * t0 + term11 * t1, qgrid, 2 * index);
                  atomic_add(term02 * t0 + term12 * t1, qgrid, 2 * index + 1);
               }
            } // end for (int iy)
         }
      } // end if (UIND)
   }
}

void gridPchg_acc(PMEUnit pme_u, real* pchg)
{
   gridPut_acc<PCHG>(pme_u, pchg, nullptr);
}

void gridDisp_acc(PMEUnit pme_u, real* csix)
{
   gridPut_acc<DISP>(pme_u, csix, nullptr);
}

void gridMpole_acc(PMEUnit pme_u, real (*fmp)[10])
{
   gridPut_acc<MPOLE>(pme_u, (real*)fmp, nullptr);
}

void gridUind_acc(PMEUnit pme_u, real (*fuind)[3], real (*fuinp)[3])
{
   gridPut_acc<UIND>(pme_u, (real*)fuind, (real*)fuinp);
}

template <bool DO_E, bool DO_V>
static void pmeConv_acc1(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
{
   auto& st = *pme_u;
   real(*restrict qgrid)[2] = reinterpret_cast<real(*)[2]>(st.qgrid);
   const real* bsmod1 = st.bsmod1;
   const real* bsmod2 = st.bsmod2;
   const real* bsmod3 = st.bsmod3;

   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const int nff = nfft1 * nfft2;
   const int ntot = nfft1 * nfft2 * nfft3;

   const real f = electric / dielec;
   real aewald = st.aewald;
   real pterm = pi / aewald;
   pterm *= pterm;
   real box_volume = boxVolume();

   auto bufsize = bufferSize();
   #pragma acc parallel loop independent\
               async(use_pme_stream ? g::qpme : g::q0)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(gpu_e,gpu_vir,qgrid,bsmod1,bsmod2,bsmod3)
   for (int i = 0; i < ntot; ++i) {
      if (i == 0) {
         qgrid[0][0] = 0;
         qgrid[0][1] = 0;
         continue;
      }

      int k3 = i / nff;
      int j = i - k3 * nff;
      int k2 = j / nfft1;
      int k1 = j - k2 * nfft1;

      int r1 = (k1 < (nfft1 + 1) / 2) ? k1 : (k1 - nfft1);
      int r2 = (k2 < (nfft2 + 1) / 2) ? k2 : (k2 - nfft2);
      int r3 = (k3 < (nfft3 + 1) / 2) ? k3 : (k3 - nfft3);

      real h1 = recipa.x * r1 + recipb.x * r2 + recipc.x * r3;
      real h2 = recipa.y * r1 + recipb.y * r2 + recipc.y * r3;
      real h3 = recipa.z * r1 + recipb.z * r2 + recipc.z * r3;
      real hsq = h1 * h1 + h2 * h2 + h3 * h3;

      real gridx = qgrid[i][0];
      real gridy = qgrid[i][1];
      real term = -pterm * hsq;
      real expterm = 0;
      if (term > -50) {
         real denom = hsq * pi * box_volume * bsmod1[k1] * bsmod2[k2] * bsmod3[k3];
         expterm = REAL_EXP(term) / denom;
         if (box_shape == BoxShape::UNBOUND)
            expterm *= (1 - REAL_COS(pi * lvec1.x * REAL_SQRT(hsq)));
         else if (box_shape == BoxShape::OCT)
            if ((k1 + k2 + k3) & 1)
               expterm = 0; // end if ((k1 + k2 + k3) % 2 != 0)

         if CONSTEXPR (DO_E || DO_V) {
            real struc2 = gridx * gridx + gridy * gridy;
            real eterm = 0.5f * f * expterm * struc2;
            if CONSTEXPR (DO_E) {
               atomic_add(eterm, gpu_e, i & (bufsize - 1));
            }
            if CONSTEXPR (DO_V) {
               real vterm = (2 / hsq) * (1 - term) * eterm;

               real vxx = (h1 * h1 * vterm - eterm);
               real vxy = h1 * h2 * vterm;
               real vxz = h1 * h3 * vterm;
               real vyy = (h2 * h2 * vterm - eterm);
               real vyz = h2 * h3 * vterm;
               real vzz = (h3 * h3 * vterm - eterm);
               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_vir, i & (bufsize - 1));
            }
         } // end if (e or v)
      }

      // complete the transformation of the PME grid
      qgrid[i][0] = gridx * expterm;
      qgrid[i][1] = gridy * expterm;
   }
}

void pmeConv_acc(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
{
   if (gpu_vir == nullptr) {
      if (gpu_e == nullptr) {
         pmeConv_acc1<false, false>(pme_u, nullptr, nullptr);
      } else {
         pmeConv_acc1<true, false>(pme_u, gpu_e, nullptr);
      }
   } else {
      if (gpu_e == nullptr) {
         pmeConv_acc1<false, true>(pme_u, nullptr, gpu_vir);
      } else {
         pmeConv_acc1<true, true>(pme_u, gpu_e, gpu_vir);
      }
   }
}

template <class T>
static void fphiGet_acc(PMEUnit pme_u, real* opt1, real* opt2, real* opt3)
{
   auto& st = *pme_u;
   auto* qgrid = st.qgrid;

   MAYBE_UNUSED real(*fphi)[20] = reinterpret_cast<real(*)[20]>(opt1);
   MAYBE_UNUSED real(*fdip_phi1)[10] = reinterpret_cast<real(*)[10]>(opt1);
   MAYBE_UNUSED real(*fdip_phi2)[10] = reinterpret_cast<real(*)[10]>(opt2);
   MAYBE_UNUSED real(*fdip_sum_phi)[20] = reinterpret_cast<real(*)[20]>(opt3);

   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const int bsorder = st.bsorder;
   assert(bsorder <= 5);

   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(fphi,fdip_phi1,fdip_phi2,fdip_sum_phi,x,y,z,qgrid)
   for (int i = 0; i < n; ++i) {
      real thetai1[4 * 5];
      real thetai2[4 * 5];
      real thetai3[4 * 5];

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];

      // map fractional coordinate w from [-0.5 + k, 0.5 + k) to [0, 1)
      // w -> (w + 0.5) - FLOOR(w + 0.5)
      // see also subroutine bspline_fill in pmestuf.f

      real w1 = xi * recipa.x + yi * recipa.y + zi * recipa.z;
      w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
      real fr1 = nfft1 * w1;
      int igrid1 = REAL_FLOOR(fr1);
      w1 = fr1 - igrid1;

      real w2 = xi * recipb.x + yi * recipb.y + zi * recipb.z;
      w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
      real fr2 = nfft2 * w2;
      int igrid2 = REAL_FLOOR(fr2);
      w2 = fr2 - igrid2;

      real w3 = xi * recipc.x + yi * recipc.y + zi * recipc.z;
      w3 = w3 + 0.5f - REAL_FLOOR(w3 + 0.5f);
      real fr3 = nfft3 * w3;
      int igrid3 = REAL_FLOOR(fr3);
      w3 = fr3 - igrid3;

      igrid1 = igrid1 - bsorder + 1;
      igrid2 = igrid2 - bsorder + 1;
      igrid3 = igrid3 - bsorder + 1;
      igrid1 += (igrid1 < 0 ? nfft1 : 0);
      igrid2 += (igrid2 < 0 ? nfft2 : 0);
      igrid3 += (igrid3 < 0 ? nfft3 : 0);

      if CONSTEXPR (eq<T, MPOLE>() || eq<T, UIND>() || eq<T, UIND2>()) {
         bsplgen<4>(w1, thetai1, bsorder);
         bsplgen<4>(w2, thetai2, bsorder);
         bsplgen<4>(w3, thetai3, bsorder);
      }

      if CONSTEXPR (eq<T, MPOLE>()) {
         real tuv000 = 0;
         real tuv001 = 0;
         real tuv010 = 0;
         real tuv100 = 0;
         real tuv200 = 0;
         real tuv020 = 0;
         real tuv002 = 0;
         real tuv110 = 0;
         real tuv101 = 0;
         real tuv011 = 0;
         real tuv300 = 0;
         real tuv030 = 0;
         real tuv003 = 0;
         real tuv210 = 0;
         real tuv201 = 0;
         real tuv120 = 0;
         real tuv021 = 0;
         real tuv102 = 0;
         real tuv012 = 0;
         real tuv111 = 0;
         #pragma acc loop seq
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            real v2 = thetai3[4 * iz + 2];
            real v3 = thetai3[4 * iz + 3];
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            real tu20 = 0;
            real tu11 = 0;
            real tu02 = 0;
            real tu30 = 0;
            real tu21 = 0;
            real tu12 = 0;
            real tu03 = 0;
            #pragma acc loop seq
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy];
               real u1 = thetai2[4 * iy + 1];
               real u2 = thetai2[4 * iy + 2];
               real u3 = thetai2[4 * iy + 3];
               real t0 = 0;
               real t1 = 0;
               real t2 = 0;
               real t3 = 0;
               #pragma acc loop seq
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  real tq = qgrid[2 * (xbase + ybase + zbase)];
                  t0 += tq * thetai1[4 * ix];
                  t1 += tq * thetai1[4 * ix + 1];
                  t2 += tq * thetai1[4 * ix + 2];
                  t3 += tq * thetai1[4 * ix + 3];
               }
               tu00 += t0 * u0;
               tu10 += t1 * u0;
               tu01 += t0 * u1;
               tu20 += t2 * u0;
               tu11 += t1 * u1;
               tu02 += t0 * u2;
               tu30 += t3 * u0;
               tu21 += t2 * u1;
               tu12 += t1 * u2;
               tu03 += t0 * u3;
            }
            tuv000 += tu00 * v0;
            tuv100 += tu10 * v0;
            tuv010 += tu01 * v0;
            tuv001 += tu00 * v1;
            tuv200 += tu20 * v0;
            tuv020 += tu02 * v0;
            tuv002 += tu00 * v2;
            tuv110 += tu11 * v0;
            tuv101 += tu10 * v1;
            tuv011 += tu01 * v1;
            tuv300 += tu30 * v0;
            tuv030 += tu03 * v0;
            tuv003 += tu00 * v3;
            tuv210 += tu21 * v0;
            tuv201 += tu20 * v1;
            tuv120 += tu12 * v0;
            tuv021 += tu02 * v1;
            tuv102 += tu10 * v2;
            tuv012 += tu01 * v2;
            tuv111 += tu11 * v1;
         }
         fphi[i][0] = tuv000;
         fphi[i][1] = tuv100;
         fphi[i][2] = tuv010;
         fphi[i][3] = tuv001;
         fphi[i][4] = tuv200;
         fphi[i][5] = tuv020;
         fphi[i][6] = tuv002;
         fphi[i][7] = tuv110;
         fphi[i][8] = tuv101;
         fphi[i][9] = tuv011;
         fphi[i][10] = tuv300;
         fphi[i][11] = tuv030;
         fphi[i][12] = tuv003;
         fphi[i][13] = tuv210;
         fphi[i][14] = tuv201;
         fphi[i][15] = tuv120;
         fphi[i][16] = tuv021;
         fphi[i][17] = tuv102;
         fphi[i][18] = tuv012;
         fphi[i][19] = tuv111;
      } // end if (fphi_mpole)

      if CONSTEXPR (eq<T, UIND>() || eq<T, UIND2>()) {
         real tuv100_1 = 0;
         real tuv010_1 = 0;
         real tuv001_1 = 0;
         real tuv200_1 = 0;
         real tuv020_1 = 0;
         real tuv002_1 = 0;
         real tuv110_1 = 0;
         real tuv101_1 = 0;
         real tuv011_1 = 0;
         real tuv100_2 = 0;
         real tuv010_2 = 0;
         real tuv001_2 = 0;
         real tuv200_2 = 0;
         real tuv020_2 = 0;
         real tuv002_2 = 0;
         real tuv110_2 = 0;
         real tuv101_2 = 0;
         real tuv011_2 = 0;
         real tuv000 = 0;
         real tuv001 = 0;
         real tuv010 = 0;
         real tuv100 = 0;
         real tuv200 = 0;
         real tuv020 = 0;
         real tuv002 = 0;
         real tuv110 = 0;
         real tuv101 = 0;
         real tuv011 = 0;
         real tuv300 = 0;
         real tuv030 = 0;
         real tuv003 = 0;
         real tuv210 = 0;
         real tuv201 = 0;
         real tuv120 = 0;
         real tuv021 = 0;
         real tuv102 = 0;
         real tuv012 = 0;
         real tuv111 = 0;
         #pragma acc loop seq
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            real v2 = thetai3[4 * iz + 2];
            real v3 = thetai3[4 * iz + 3];
            real tu00_1 = 0;
            real tu01_1 = 0;
            real tu10_1 = 0;
            real tu20_1 = 0;
            real tu11_1 = 0;
            real tu02_1 = 0;
            real tu00_2 = 0;
            real tu01_2 = 0;
            real tu10_2 = 0;
            real tu20_2 = 0;
            real tu11_2 = 0;
            real tu02_2 = 0;
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            real tu20 = 0;
            real tu11 = 0;
            real tu02 = 0;
            real tu30 = 0;
            real tu21 = 0;
            real tu12 = 0;
            real tu03 = 0;
            #pragma acc loop seq
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy];
               real u1 = thetai2[4 * iy + 1];
               real u2 = thetai2[4 * iy + 2];
               real u3 = thetai2[4 * iy + 3];
               real t0_1 = 0;
               real t1_1 = 0;
               real t2_1 = 0;
               real t0_2 = 0;
               real t1_2 = 0;
               real t2_2 = 0;
               real t3 = 0;
               #pragma acc loop seq
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  real tq_1 = qgrid[2 * (xbase + ybase + zbase)];
                  real tq_2 = qgrid[2 * (xbase + ybase + zbase) + 1];
                  t0_1 += tq_1 * thetai1[4 * ix];
                  t1_1 += tq_1 * thetai1[4 * ix + 1];
                  t2_1 += tq_1 * thetai1[4 * ix + 2];
                  t0_2 += tq_2 * thetai1[4 * ix];
                  t1_2 += tq_2 * thetai1[4 * ix + 1];
                  t2_2 += tq_2 * thetai1[4 * ix + 2];
                  t3 += (tq_1 + tq_2) * thetai1[4 * ix + 3];
               }
               tu00_1 += t0_1 * u0;
               tu10_1 += t1_1 * u0;
               tu01_1 += t0_1 * u1;
               tu20_1 += t2_1 * u0;
               tu11_1 += t1_1 * u1;
               tu02_1 += t0_1 * u2;
               tu00_2 += t0_2 * u0;
               tu10_2 += t1_2 * u0;
               tu01_2 += t0_2 * u1;
               tu20_2 += t2_2 * u0;
               tu11_2 += t1_2 * u1;
               tu02_2 += t0_2 * u2;
               real t0 = t0_1 + t0_2;
               real t1 = t1_1 + t1_2;
               real t2 = t2_1 + t2_2;
               tu00 += t0 * u0;
               tu10 += t1 * u0;
               tu01 += t0 * u1;
               tu20 += t2 * u0;
               tu11 += t1 * u1;
               tu02 += t0 * u2;
               tu30 += t3 * u0;
               tu21 += t2 * u1;
               tu12 += t1 * u2;
               tu03 += t0 * u3;
            }
            tuv100_1 += tu10_1 * v0;
            tuv010_1 += tu01_1 * v0;
            tuv001_1 += tu00_1 * v1;
            tuv200_1 += tu20_1 * v0;
            tuv020_1 += tu02_1 * v0;
            tuv002_1 += tu00_1 * v2;
            tuv110_1 += tu11_1 * v0;
            tuv101_1 += tu10_1 * v1;
            tuv011_1 += tu01_1 * v1;
            tuv100_2 += tu10_2 * v0;
            tuv010_2 += tu01_2 * v0;
            tuv001_2 += tu00_2 * v1;
            tuv200_2 += tu20_2 * v0;
            tuv020_2 += tu02_2 * v0;
            tuv002_2 += tu00_2 * v2;
            tuv110_2 += tu11_2 * v0;
            tuv101_2 += tu10_2 * v1;
            tuv011_2 += tu01_2 * v1;
            tuv000 += tu00 * v0;
            tuv100 += tu10 * v0;
            tuv010 += tu01 * v0;
            tuv001 += tu00 * v1;
            tuv200 += tu20 * v0;
            tuv020 += tu02 * v0;
            tuv002 += tu00 * v2;
            tuv110 += tu11 * v0;
            tuv101 += tu10 * v1;
            tuv011 += tu01 * v1;
            tuv300 += tu30 * v0;
            tuv030 += tu03 * v0;
            tuv003 += tu00 * v3;
            tuv210 += tu21 * v0;
            tuv201 += tu20 * v1;
            tuv120 += tu12 * v0;
            tuv021 += tu02 * v1;
            tuv102 += tu10 * v2;
            tuv012 += tu01 * v2;
            tuv111 += tu11 * v1;
         } // end for (iz)
         fdip_phi1[i][0] = 0;
         fdip_phi1[i][1] = tuv100_1;
         fdip_phi1[i][2] = tuv010_1;
         fdip_phi1[i][3] = tuv001_1;
         fdip_phi1[i][4] = tuv200_1;
         fdip_phi1[i][5] = tuv020_1;
         fdip_phi1[i][6] = tuv002_1;
         fdip_phi1[i][7] = tuv110_1;
         fdip_phi1[i][8] = tuv101_1;
         fdip_phi1[i][9] = tuv011_1;

         fdip_phi2[i][0] = 0;
         fdip_phi2[i][1] = tuv100_2;
         fdip_phi2[i][2] = tuv010_2;
         fdip_phi2[i][3] = tuv001_2;
         fdip_phi2[i][4] = tuv200_2;
         fdip_phi2[i][5] = tuv020_2;
         fdip_phi2[i][6] = tuv002_2;
         fdip_phi2[i][7] = tuv110_2;
         fdip_phi2[i][8] = tuv101_2;
         fdip_phi2[i][9] = tuv011_2;

         if CONSTEXPR (eq<T, UIND>()) {
            fdip_sum_phi[i][0] = tuv000;
            fdip_sum_phi[i][1] = tuv100;
            fdip_sum_phi[i][2] = tuv010;
            fdip_sum_phi[i][3] = tuv001;
            fdip_sum_phi[i][4] = tuv200;
            fdip_sum_phi[i][5] = tuv020;
            fdip_sum_phi[i][6] = tuv002;
            fdip_sum_phi[i][7] = tuv110;
            fdip_sum_phi[i][8] = tuv101;
            fdip_sum_phi[i][9] = tuv011;
            fdip_sum_phi[i][10] = tuv300;
            fdip_sum_phi[i][11] = tuv030;
            fdip_sum_phi[i][12] = tuv003;
            fdip_sum_phi[i][13] = tuv210;
            fdip_sum_phi[i][14] = tuv201;
            fdip_sum_phi[i][15] = tuv120;
            fdip_sum_phi[i][16] = tuv021;
            fdip_sum_phi[i][17] = tuv102;
            fdip_sum_phi[i][18] = tuv012;
            fdip_sum_phi[i][19] = tuv111;
         }
      } // end if (fphi_uind)
   }    // end for (int i)
}

void fphiMpole_acc(PMEUnit pme_u, real (*gpu_fphi)[20])
{
   fphiGet_acc<MPOLE>(pme_u, (real*)gpu_fphi, nullptr, nullptr);
}

void fphiUind_acc(PMEUnit pme_u, real (*gpu_fdip_phi1)[10], real (*gpu_fdip_phi2)[10], real (*gpu_fdip_sum_phi)[20])
{
   fphiGet_acc<UIND>(pme_u, (real*)gpu_fdip_phi1, (real*)gpu_fdip_phi2, (real*)gpu_fdip_sum_phi);
}

void fphiUind2_acc(PMEUnit pme_u, real (*gpu_fdip_phi1)[10], real (*gpu_fdip_phi2)[10])
{
   fphiGet_acc<UIND2>(pme_u, (real*)gpu_fdip_phi1, (real*)gpu_fdip_phi2, nullptr);
}

void rpoleToCmp_acc()
{
   #pragma acc parallel loop independent async deviceptr(rpole,cmp)
   for (int i = 0; i < n; ++i) {
      cmp[i][0] = rpole[i][MPL_PME_0];
      cmp[i][1] = rpole[i][MPL_PME_X];
      cmp[i][2] = rpole[i][MPL_PME_Y];
      cmp[i][3] = rpole[i][MPL_PME_Z];
      cmp[i][4] = rpole[i][MPL_PME_XX];
      cmp[i][5] = rpole[i][MPL_PME_YY];
      cmp[i][6] = rpole[i][MPL_PME_ZZ];
      cmp[i][7] = 2 * rpole[i][MPL_PME_XY];
      cmp[i][8] = 2 * rpole[i][MPL_PME_XZ];
      cmp[i][9] = 2 * rpole[i][MPL_PME_YZ];
   }
}

void cmpToFmp_acc(PMEUnit pme_u, const real (*cmp)[10], real (*fmp)[10])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   #pragma acc parallel loop independent async deviceptr(cmp,fmp)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)
   for (int iatom = 0; iatom < n; ++iatom) {
      real a[3][3];
      // see also subroutine cart_to_frac in pmestuf.f
      // set the reciprocal vector transformation matrix
      a[0][0] = nfft1 * recipa.x;
      a[0][1] = nfft2 * recipb.x;
      a[0][2] = nfft3 * recipc.x;
      a[1][0] = nfft1 * recipa.y;
      a[1][1] = nfft2 * recipb.y;
      a[1][2] = nfft3 * recipc.y;
      a[2][0] = nfft1 * recipa.z;
      a[2][1] = nfft2 * recipb.z;
      a[2][2] = nfft3 * recipc.z;

      // data qi1  / 1, 2, 3, 1, 1, 2 /
      // data qi2  / 1, 2, 3, 2, 3, 3 /
      constexpr int qi1[] = {0, 1, 2, 0, 0, 1};
      constexpr int qi2[] = {0, 1, 2, 1, 2, 2};
      real ctf6[6][6];
      // get the Cartesian to fractional conversion matrix
      #pragma acc loop seq
      for (int i1 = 0; i1 < 3; ++i1) {
         int k = qi1[i1];
         #pragma acc loop seq
         for (int i2 = 0; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ctf6[i2][i1] = a[i][k] * a[j][k];
         }
      }
      #pragma acc loop seq
      for (int i1 = 3; i1 < 6; ++i1) {
         int k = qi1[i1];
         int m = qi2[i1];
         #pragma acc loop seq
         for (int i2 = 0; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ctf6[i2][i1] = a[i][k] * a[j][m] + a[j][k] * a[i][m];
         }
      }

      // apply the transformation to get the fractional multipoles
      real cmpi[10], fmpi[10];
      cmpi[0] = cmp[iatom][0];
      cmpi[1] = cmp[iatom][1];
      cmpi[2] = cmp[iatom][2];
      cmpi[3] = cmp[iatom][3];
      cmpi[4] = cmp[iatom][4];
      cmpi[5] = cmp[iatom][5];
      cmpi[6] = cmp[iatom][6];
      cmpi[7] = cmp[iatom][7];
      cmpi[8] = cmp[iatom][8];
      cmpi[9] = cmp[iatom][9];

      fmpi[0] = cmpi[0];
      #pragma acc loop seq
      for (int j = 1; j < 4; ++j) {
         fmpi[j] = 0;
         #pragma acc loop seq
         for (int k = 1; k < 4; ++k) {
            fmpi[j] += a[k - 1][j - 1] * cmpi[k];
         }
      }
      #pragma acc loop seq
      for (int j = 4; j < 10; ++j) {
         fmpi[j] = 0;
         #pragma acc loop seq
         for (int k = 4; k < 10; ++k) {
            fmpi[j] += ctf6[k - 4][j - 4] * cmpi[k];
         }
      }

      fmp[iatom][0] = fmpi[0];
      fmp[iatom][1] = fmpi[1];
      fmp[iatom][2] = fmpi[2];
      fmp[iatom][3] = fmpi[3];
      fmp[iatom][4] = fmpi[4];
      fmp[iatom][5] = fmpi[5];
      fmp[iatom][6] = fmpi[6];
      fmp[iatom][7] = fmpi[7];
      fmp[iatom][8] = fmpi[8];
      fmp[iatom][9] = fmpi[9];
   }
}

void cuindToFuind_acc(PMEUnit pme_u, const real (*cind)[3], const real (*cinp)[3], //
   real (*fuind)[3], real (*fuinp)[3])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   #pragma acc parallel loop independent async deviceptr(cind,cinp,fuind,fuinp)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)
   for (int i = 0; i < n; ++i) {
      real a[3][3];
      a[0][0] = nfft1 * recipa.x;
      a[0][1] = nfft2 * recipb.x;
      a[0][2] = nfft3 * recipc.x;
      a[1][0] = nfft1 * recipa.y;
      a[1][1] = nfft2 * recipb.y;
      a[1][2] = nfft3 * recipc.y;
      a[2][0] = nfft1 * recipa.z;
      a[2][1] = nfft2 * recipb.z;
      a[2][2] = nfft3 * recipc.z;

      #pragma acc loop seq
      for (int j = 0; j < 3; ++j) {
         fuind[i][j] = a[0][j] * cind[i][0] + a[1][j] * cind[i][1] + a[2][j] * cind[i][2];
         fuinp[i][j] = a[0][j] * cinp[i][0] + a[1][j] * cinp[i][1] + a[2][j] * cinp[i][2];
      }
   }
}

void fphiToCphi_acc(PMEUnit pme_u, const real (*fphi)[20], real (*cphi)[10])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   #pragma acc parallel loop async deviceptr(fphi,cphi)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)
   for (int iatom = 0; iatom < n; ++iatom) {
      real a[3][3];
      // see also subroutine frac_to_cart in pmestuf.f
      // set the reciprocal vector transformation matrix
      a[0][0] = nfft1 * recipa.x;
      a[1][0] = nfft2 * recipb.x;
      a[2][0] = nfft3 * recipc.x;
      a[0][1] = nfft1 * recipa.y;
      a[1][1] = nfft2 * recipb.y;
      a[2][1] = nfft3 * recipc.y;
      a[0][2] = nfft1 * recipa.z;
      a[1][2] = nfft2 * recipb.z;
      a[2][2] = nfft3 * recipc.z;

      // data qi1  / 1, 2, 3, 1, 1, 2 /
      // data qi2  / 1, 2, 3, 2, 3, 3 /
      constexpr int qi1[] = {0, 1, 2, 0, 0, 1};
      constexpr int qi2[] = {0, 1, 2, 1, 2, 2};

      real ftc6[6][6];
      // get the fractional to Cartesian conversion matrix
      #pragma acc loop seq
      for (int i1 = 0; i1 < 3; ++i1) {
         int k = qi1[i1];
         #pragma acc loop seq
         for (int i2 = 0; i2 < 3; ++i2) {
            int i = qi1[i2];
            ftc6[i2][i1] = a[i][k] * a[i][k];
         }
         #pragma acc loop seq
         for (int i2 = 3; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ftc6[i2][i1] = 2 * a[i][k] * a[j][k];
         }
      }

      #pragma acc loop seq
      for (int i1 = 3; i1 < 6; ++i1) {
         int k = qi1[i1];
         int m = qi2[i1];
         #pragma acc loop seq
         for (int i2 = 0; i2 < 3; ++i2) {
            int i = qi1[i2];
            ftc6[i2][i1] = a[i][k] * a[i][m];
         }
         #pragma acc loop seq
         for (int i2 = 3; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ftc6[i2][i1] = a[i][k] * a[j][m] + a[i][m] * a[j][k];
         }
      }

      // apply the transformation to get the Cartesian potential
      real fphii[10], cphii[10];
      fphii[0] = fphi[iatom][0];
      fphii[1] = fphi[iatom][1];
      fphii[2] = fphi[iatom][2];
      fphii[3] = fphi[iatom][3];
      fphii[4] = fphi[iatom][4];
      fphii[5] = fphi[iatom][5];
      fphii[6] = fphi[iatom][6];
      fphii[7] = fphi[iatom][7];
      fphii[8] = fphi[iatom][8];
      fphii[9] = fphi[iatom][9];

      cphii[0] = fphii[0];
      #pragma acc loop seq
      for (int j = 1; j < 4; ++j) {
         cphii[j] = 0;
         #pragma acc loop seq
         for (int k = 1; k < 4; ++k) {
            cphii[j] += a[k - 1][j - 1] * fphii[k];
         }
      }
      #pragma acc loop seq
      for (int j = 4; j < 10; ++j) {
         cphii[j] = 0;
         #pragma acc loop seq
         for (int k = 4; k < 10; ++k) {
            cphii[j] += ftc6[k - 4][j - 4] * fphii[k];
         }
      }

      cphi[iatom][0] = cphii[0];
      cphi[iatom][1] = cphii[1];
      cphi[iatom][2] = cphii[2];
      cphi[iatom][3] = cphii[3];
      cphi[iatom][4] = cphii[4];
      cphi[iatom][5] = cphii[5];
      cphi[iatom][6] = cphii[6];
      cphi[iatom][7] = cphii[7];
      cphi[iatom][8] = cphii[8];
      cphi[iatom][9] = cphii[9];
   }
}
}
