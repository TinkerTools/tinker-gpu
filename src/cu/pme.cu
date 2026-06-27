#include "ff/amoeba/empole.h"
#include "ff/box.h"
#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/bsplgen.h"
#include "seq/launch.h"
#include "tool/error.h"

namespace tinker {
// compute theta values on the fly
template <class T, int bsorder>
__global__
void gridPut_cu1(const real* restrict x, const real* restrict y, const real* restrict z, int n, int nfft1, int nfft2,
   int nfft3, const real* restrict ptr1, const real* restrict ptr2, real* restrict qgrid, real3 recip_a, real3 recip_b,
   real3 recip_c)
{
   real thetai1[4 * 5];
   real thetai2[4 * 5];
   real thetai3[4 * 5];
   __shared__ real sharedarray[5 * 5 * PME_BLOCKDIM];
   real* restrict array = &sharedarray[5 * 5 * threadIdx.x];

   MAYBE_UNUSED const real* pchg = ptr1;
   MAYBE_UNUSED const real(*fmp)[10] = (real(*)[10])ptr1;
   MAYBE_UNUSED const real(*fuind)[3] = (real(*)[3])ptr1;
   MAYBE_UNUSED const real(*fuinp)[3] = (real(*)[3])ptr2;

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];

      real w1 = xi * recip_a.x + yi * recip_a.y + zi * recip_a.z;
      w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
      real fr1 = nfft1 * w1;
      int igrid1 = REAL_FLOOR(fr1);
      w1 = fr1 - igrid1;

      real w2 = xi * recip_b.x + yi * recip_b.y + zi * recip_b.z;
      w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
      real fr2 = nfft2 * w2;
      int igrid2 = REAL_FLOOR(fr2);
      w2 = fr2 - igrid2;

      real w3 = xi * recip_c.x + yi * recip_c.y + zi * recip_c.z;
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
         real chgi = pchg[i];
         if (chgi == 0)
            continue;

         bsplgen<1, bsorder>(w1, thetai1, array);
         bsplgen<1, bsorder>(w2, thetai2, array);
         bsplgen<1, bsorder>(w3, thetai3, array);

         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz] * chgi;
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy] * v0;
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real term = thetai1[4 * ix] * u0;
                  atomic_add(term, qgrid, 2 * index);
               }
            }
         }
      } // end if (PCHG or DISP)

      if CONSTEXPR (eq<T, MPOLE>()) {
         bsplgen<3, bsorder>(w1, thetai1, array);
         bsplgen<3, bsorder>(w2, thetai2, array);
         bsplgen<3, bsorder>(w3, thetai3, array);

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
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            real v2 = thetai3[4 * iz + 2];
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
         bsplgen<2, bsorder>(w1, thetai1, array);
         bsplgen<2, bsorder>(w2, thetai2, array);
         bsplgen<2, bsorder>(w3, thetai3, array);

         real fuindi0 = fuind[i][0];
         real fuindi1 = fuind[i][1];
         real fuindi2 = fuind[i][2];
         real fuinpi0 = fuinp[i][0];
         real fuinpi1 = fuinp[i][1];
         real fuinpi2 = fuinp[i][2];
         #pragma unroll
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            #pragma unroll
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
               #pragma unroll
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

// use pre-computed theta values
template <class T, int bsorder>
__global__
void gridPut_cu2(const int* restrict igrid, const real* restrict thetai1, const real* restrict thetai2,
   const real* restrict thetai3, const Spatial::SortedAtom* restrict sorted, int n, int padded_n, int nfft1, int nfft2,
   int nfft3, const real* restrict ptr1, const real* ptr2, real* restrict qgrid)
{
   constexpr int bso2 = bsorder * bsorder;
   constexpr int bso3 = bsorder * bso2;
   for (int m = threadIdx.x + blockIdx.x * blockDim.x; m < n * bso3; m += blockDim.x * gridDim.x) {
      // m = i0 * bso3 + j;
      int i0 = m / bso3;
      int j = m - i0 * bso3;

      int i = sorted[i0].unsorted;
      int igrid1 = igrid[3 * i + 0];
      int igrid2 = igrid[3 * i + 1];
      int igrid3 = igrid[3 * i + 2];

      int iz = j / bso2;
      j -= iz * bso2;
      int iy = j / bsorder;
      int ix = j - (j / bsorder) * bsorder;

      real v0 = thetai3[(4 * iz + 0) * padded_n + i];
      real v1 = thetai3[(4 * iz + 1) * padded_n + i];
      int zbase = igrid3 + iz;
      zbase -= (zbase >= nfft3 ? nfft3 : 0);
      zbase *= (nfft1 * nfft2);

      real u0 = thetai2[(4 * iy + 0) * padded_n + i];
      real u1 = thetai2[(4 * iy + 1) * padded_n + i];
      int ybase = igrid2 + iy;
      ybase -= (ybase >= nfft2 ? nfft2 : 0);
      ybase *= nfft1;

      real t0 = thetai1[(4 * ix + 0) * padded_n + i];
      real t1 = thetai1[(4 * ix + 1) * padded_n + i];
      int xbase = igrid1 + ix;
      xbase -= (xbase >= nfft1 ? nfft1 : 0);
      int index = xbase + ybase + zbase;

      if CONSTEXPR (eq<T, MPOLE>()) {
         real v2 = thetai3[(4 * iz + 2) * padded_n + i];
         real u2 = thetai2[(4 * iy + 2) * padded_n + i];
         real t2 = thetai1[(4 * ix + 2) * padded_n + i];
         real fmpi0 = ptr1[i * 10 + MPL_PME_0];
         real fmpix = ptr1[i * 10 + MPL_PME_X];
         real fmpiy = ptr1[i * 10 + MPL_PME_Y];
         real fmpiz = ptr1[i * 10 + MPL_PME_Z];
         real fmpixx = ptr1[i * 10 + MPL_PME_XX];
         real fmpiyy = ptr1[i * 10 + MPL_PME_YY];
         real fmpizz = ptr1[i * 10 + MPL_PME_ZZ];
         real fmpixy = ptr1[i * 10 + MPL_PME_XY];
         real fmpixz = ptr1[i * 10 + MPL_PME_XZ];
         real fmpiyz = ptr1[i * 10 + MPL_PME_YZ];
         real term0 = fmpi0 * u0 * v0 + fmpiy * u1 * v0 + fmpiz * u0 * v1 + fmpiyy * u2 * v0 + fmpizz * u0 * v2
            + fmpiyz * u1 * v1;
         real term1 = fmpix * u0 * v0 + fmpixy * u1 * v0 + fmpixz * u0 * v1;
         real term2 = fmpixx * u0 * v0;
         atomic_add(term0 * t0 + term1 * t1 + term2 * t2, qgrid, 2 * index);
      }

      if CONSTEXPR (eq<T, UIND>()) {
         real3 fd = make_real3(ptr1[3 * i + 0], ptr1[3 * i + 1], ptr1[3 * i + 2]);
         real3 fp = make_real3(ptr2[3 * i + 0], ptr2[3 * i + 1], ptr2[3 * i + 2]);
         real3 tuv = make_real3(t1 * u0 * v0, t0 * u1 * v0, t0 * u0 * v1);
         atomic_add(dot3(fd, tuv), qgrid, 2 * index);
         atomic_add(dot3(fp, tuv), qgrid, 2 * index + 1);
      }
   }
}

void gridPchg_cu(PMEUnit pme_u, real* pchg)
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;
   int nt = n1 * n2 * n3;

   auto stream = g::s0;
   if (use_pme_stream)
      stream = g::spme;
   using type = std::remove_pointer<decltype(st.qgrid)>::type;
   check_rt(cudaMemsetAsync(st.qgrid, 0, 2 * nt * sizeof(type), stream));
   if (st.bsorder == 5) {
      auto ker = gridPut_cu1<PCHG, 5>;
      launch_k2s(stream, PME_BLOCKDIM, n, ker, x, y, z, n, n1, n2, n3, pchg, nullptr, st.qgrid, recipa, recipb, recipc);
   } else if (st.bsorder == 4) {
      auto ker = gridPut_cu1<PCHG, 4>;
      launch_k2s(stream, PME_BLOCKDIM, n, ker, x, y, z, n, n1, n2, n3, pchg, nullptr, st.qgrid, recipa, recipb, recipc);
   }
}

void gridDisp_cu(PMEUnit pme_u, real* csix)
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;
   int nt = n1 * n2 * n3;

   darray::zero(g::q0, 2 * nt, st.qgrid);
   auto ker = gridPut_cu1<DISP, 4>;
   launch_k2s(g::s0, PME_BLOCKDIM, n, ker, x, y, z, n, n1, n2, n3, csix, nullptr, st.qgrid, recipa, recipb, recipc);
}

void gridMpole_cu(PMEUnit pme_u, real (*fmp)[10])
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;
   int nt = n1 * n2 * n3;

   darray::zero(g::q0, 2 * nt, st.qgrid);
   if (TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE) {
      auto ker = gridPut_cu1<MPOLE, 5>;
      launch_k2s(g::s0, PME_BLOCKDIM, n, ker, x, y, z, n, n1, n2, n3, (const real*)fmp, nullptr, st.qgrid, recipa,
         recipb, recipc);
   } else {
      auto ker = gridPut_cu2<MPOLE, 5>;
      int npa = 5 * 5 * 5 * n;
      launch_k1s(g::s0, npa, ker, st.igrid, st.thetai1, st.thetai2, st.thetai3, mspatial_v2_unit->sorted, n, padded_n,
         n1, n2, n3, (const real*)fmp, nullptr, st.qgrid);
   }
}

void gridUind_cu(PMEUnit pme_u, real (*fuind)[3], real (*fuinp)[3])
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;
   int nt = n1 * n2 * n3;

   darray::zero(g::q0, 2 * nt, st.qgrid);
   if (TINKER_CU_THETA_ON_THE_FLY_GRID_UIND) {
      auto ker = gridPut_cu1<UIND, 5>;
      launch_k2s(g::s0, PME_BLOCKDIM, n, ker, x, y, z, n, n1, n2, n3, (const real*)fuind, (const real*)fuinp, st.qgrid,
         recipa, recipb, recipc);
   } else {
      auto ker = gridPut_cu2<UIND, 5>;
      int npa = 5 * 5 * 5 * n;
      launch_k1s(g::s0, npa, ker, st.igrid, st.thetai1, st.thetai2, st.thetai3, mspatial_v2_unit->sorted, n, padded_n,
         n1, n2, n3, (const real*)fuind, (const real*)fuinp, st.qgrid);
   }
}

template <int LEVEL, int bsorder>
__global__
void bsplineFill_cu1(int* restrict igrid, real* restrict thetai1, real* restrict thetai2, real* restrict thetai3,
   const real* restrict x, const real* restrict y, const real* restrict z, int n, int padded_n, int nfft1, int nfft2,
   int nfft3, real3 recip_a, real3 recip_b, real3 recip_c)
{
   const int nfft4[3] = {nfft1, nfft2, nfft3};
   const real3 recip4[3] = {recip_a, recip_b, recip_c};
   real* const thetai[3] = {thetai1, thetai2, thetai3};
   real array[5 * 5];

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      int igridi[3];
      for (int j = 0; j < 3; ++j) {
         real w4 = xi * recip4[j].x + yi * recip4[j].y + zi * recip4[j].z;
         w4 = w4 + 0.5f - REAL_FLOOR(w4 + 0.5f);
         real fr4 = nfft4[j] * w4;
         int igrid4 = REAL_FLOOR(fr4);
         w4 = fr4 - igrid4;
         igrid4 = igrid4 - bsorder + 1;
         igrid4 += (igrid4 < 0 ? nfft4[j] : 0);
         // write output
         igridi[j] = igrid4;
         bsplgen2<LEVEL, bsorder>(w4, thetai[j], i, padded_n, array);
      }
      igrid[3 * i + 0] = igridi[0];
      igrid[3 * i + 1] = igridi[1];
      igrid[3 * i + 2] = igridi[2];
   }
}

void bsplineFill_cu(PMEUnit u, int level)
{
   auto& st = *u;
   if (level == 2) {
      auto ker = bsplineFill_cu1<2, 5>;
      launch_k1s(g::s0, n, ker, st.igrid, st.thetai1, st.thetai2, st.thetai3, x, y, z, n, padded_n, st.nfft1, st.nfft2,
         st.nfft3, recipa, recipb, recipc);
   } else if (level == 3) {
      auto ker = bsplineFill_cu1<3, 5>;
      launch_k1s(g::s0, n, ker, st.igrid, st.thetai1, st.thetai2, st.thetai3, x, y, z, n, padded_n, st.nfft1, st.nfft2,
         st.nfft3, recipa, recipb, recipc);
   }
}

// compute theta values on the fly
template <class T, int bsorder>
__global__
void fphiGet_cu(int n, int nfft1, int nfft2, int nfft3, const real* restrict x, const real* restrict y,
   const real* restrict z, real* restrict opt1, real* restrict opt2, real* restrict opt3, const real* restrict qgrid,
   real3 recip_a, real3 recip_b, real3 recip_c)
{
   real thetai1[4 * 5];
   real thetai2[4 * 5];
   real thetai3[4 * 5];
   real array[5 * 5];

   MAYBE_UNUSED real(*restrict fphi)[20] = (real(*)[20])opt1;
   MAYBE_UNUSED real(*restrict fdip_phi1)[10] = (real(*)[10])opt1;
   MAYBE_UNUSED real(*restrict fdip_phi2)[10] = (real(*)[10])opt2;
   MAYBE_UNUSED real(*restrict fdip_sum_phi)[20] = (real(*)[20])opt3;

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];

      real w1 = xi * recip_a.x + yi * recip_a.y + zi * recip_a.z;
      w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
      real fr1 = nfft1 * w1;
      int igrid1 = REAL_FLOOR(fr1);
      w1 = fr1 - igrid1;

      real w2 = xi * recip_b.x + yi * recip_b.y + zi * recip_b.z;
      w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
      real fr2 = nfft2 * w2;
      int igrid2 = REAL_FLOOR(fr2);
      w2 = fr2 - igrid2;

      real w3 = xi * recip_c.x + yi * recip_c.y + zi * recip_c.z;
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
         bsplgen<4, bsorder>(w1, thetai1, array);
         bsplgen<4, bsorder>(w2, thetai2, array);
         bsplgen<4, bsorder>(w3, thetai3, array);
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
      }

      if CONSTEXPR (eq<T, UIND>()) {
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

      if CONSTEXPR (eq<T, UIND2>()) {
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
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real v0 = thetai3[4 * iz];
            real v1 = thetai3[4 * iz + 1];
            real v2 = thetai3[4 * iz + 2];
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
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real u0 = thetai2[4 * iy];
               real u1 = thetai2[4 * iy + 1];
               real u2 = thetai2[4 * iy + 2];
               real t0_1 = 0;
               real t1_1 = 0;
               real t2_1 = 0;
               real t0_2 = 0;
               real t1_2 = 0;
               real t2_2 = 0;
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
      }
   }
}

void fphiMpole_cu(PMEUnit pme_u, real (*fphi)[20])
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;

   auto ker = fphiGet_cu<MPOLE, 5>;
   launch_k2s(g::s0, PME_BLOCKDIM, n, ker, n, n1, n2, n3, x, y, z, (real*)fphi, nullptr, nullptr, st.qgrid, recipa,
      recipb, recipc);
}

void fphiUind_cu(PMEUnit pme_u, real (*fdip_phi1)[10], real (*fdip_phi2)[10], real (*fdip_sum_phi)[20])
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;

   auto ker = fphiGet_cu<UIND, 5>;
   launch_k2s(g::s0, PME_BLOCKDIM, n, ker, n, n1, n2, n3, x, y, z, (real*)fdip_phi1, (real*)fdip_phi2,
      (real*)fdip_sum_phi, st.qgrid, recipa, recipb, recipc);
}

void fphiUind2_cu(PMEUnit pme_u, real (*fdip_phi1)[10], real (*fdip_phi2)[10])
{
   auto& st = *pme_u;
   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;

   auto ker = fphiGet_cu<UIND2, 5>;
   launch_k2s(g::s0, PME_BLOCKDIM, n, ker, n, n1, n2, n3, x, y, z, (real*)fdip_phi1, (real*)fdip_phi2, nullptr,
      st.qgrid, recipa, recipb, recipc);
}

template <bool DO_E, bool DO_V>
__global__
static void pmeConv_cu1(int nfft1, int nfft2, int nfft3, real (*restrict qgrid)[2], const real* restrict bsmod1,
   const real* restrict bsmod2, const real* restrict bsmod3, real f, real aewald, TINKER_IMAGE_PARAMS, real box_volume,
   EnergyBuffer restrict gpu_e, VirialBuffer restrict gpu_vir)
{
   int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = blockDim.x * gridDim.x;

   int nff = nfft1 * nfft2;
   int ntot = nfft1 * nfft2 * nfft3;
   real pterm = pi / aewald;
   pterm *= pterm;

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl;
   if CONSTEXPR (DO_E) {
      ectl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz;
   if CONSTEXPR (DO_V) {
      vctlxx = 0;
      vctlyx = 0;
      vctlzx = 0;
      vctlyy = 0;
      vctlzy = 0;
      vctlzz = 0;
   }

   for (int i = ithread; i < ntot; i += stride) {
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

         if CONSTEXPR (DO_E or DO_V) {
            real struc2 = gridx * gridx + gridy * gridy;
            real eterm = 0.5f * f * expterm * struc2;
            if CONSTEXPR (DO_E) {
               ectl += floatTo<ebuf_prec>(eterm);
            }
            if CONSTEXPR (DO_V) {
               real vterm = (2 / hsq) * (1 - term) * eterm;
               real vxx = (h1 * h1 * vterm - eterm);
               real vyx = h1 * h2 * vterm;
               real vzx = h1 * h3 * vterm;
               real vyy = (h2 * h2 * vterm - eterm);
               real vzy = h2 * h3 * vterm;
               real vzz = (h3 * h3 * vterm - eterm);
               vctlxx += floatTo<vbuf_prec>(vxx);
               vctlyx += floatTo<vbuf_prec>(vyx);
               vctlzx += floatTo<vbuf_prec>(vzx);
               vctlyy += floatTo<vbuf_prec>(vyy);
               vctlzy += floatTo<vbuf_prec>(vzy);
               vctlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (e or v)
      }

      // complete the transformation of the PME grid
      qgrid[i][0] = gridx * expterm;
      qgrid[i][1] = gridy * expterm;
   }

   if CONSTEXPR (DO_E) {
      atomic_add(ectl, gpu_e, ithread);
   }
   if CONSTEXPR (DO_V) {
      atomic_add(vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz, gpu_vir, ithread);
   }
}

template <bool DO_E, bool DO_V>
static void pmeConv_cu2(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
{
   auto& st = *pme_u;
   real(*restrict qgrid)[2] = reinterpret_cast<real(*)[2]>(st.qgrid);
   const real* bsmod1 = st.bsmod1;
   const real* bsmod2 = st.bsmod2;
   const real* bsmod3 = st.bsmod3;

   int n1 = st.nfft1;
   int n2 = st.nfft2;
   int n3 = st.nfft3;

   real f = electric / dielec;
   real aewald = st.aewald;
   real box_volume = boxVolume();

   auto ker = pmeConv_cu1<DO_E, DO_V>;
   auto stream = use_pme_stream ? g::spme : g::s0;
   int ngrid = gpuGridSize(BLOCK_DIM);
   ker<<<ngrid, BLOCK_DIM, 0, stream>>>(n1, n2, n3, qgrid, bsmod1, bsmod2, bsmod3, f, aewald, TINKER_IMAGE_ARGS,
      box_volume, gpu_e, gpu_vir);
}

void pmeConv_cu(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
{
   if (gpu_vir == nullptr) {
      if (gpu_e == nullptr) {
         pmeConv_cu2<false, false>(pme_u, nullptr, nullptr);
      } else {
         pmeConv_cu2<true, false>(pme_u, gpu_e, nullptr);
      }
   } else {
      if (gpu_e == nullptr) {
         pmeConv_cu2<false, true>(pme_u, nullptr, gpu_vir);
      } else {
         pmeConv_cu2<true, true>(pme_u, gpu_e, gpu_vir);
      }
   }
}
}

namespace tinker {
__global__
void rpoleToCmp_cu1(int n, real (*restrict cmp)[10], const real (*restrict rpole)[MPL_TOTAL])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
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

void rpoleToCmp_cu()
{
   launch_k1s(g::s0, n, rpoleToCmp_cu1, n, cmp, rpole);
}

__global__
void cmpToFmp_cu1(int n, const real (*restrict cmp)[10], real (*restrict fmp)[10], int nfft1, int nfft2, int nfft3,
   TINKER_IMAGE_PARAMS)
{
   // see also subroutine cart_to_frac in pmestuf.f
   // set the reciprocal vector transformation matrix
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

   for (int iatom = ITHREAD; iatom < n; iatom += STRIDE) {
      // data qi1  / 1, 2, 3, 1, 1, 2 /
      // data qi2  / 1, 2, 3, 2, 3, 3 /
      constexpr int qi1[] = {0, 1, 2, 0, 0, 1};
      constexpr int qi2[] = {0, 1, 2, 1, 2, 2};
      real ctf6[6][6];
      // get the Cartesian to fractional conversion matrix
      for (int i1 = 0; i1 < 3; ++i1) {
         int k = qi1[i1];
         for (int i2 = 0; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ctf6[i2][i1] = a[i][k] * a[j][k];
         }
      }
      for (int i1 = 3; i1 < 6; ++i1) {
         int k = qi1[i1];
         int m = qi2[i1];
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
      for (int j = 1; j < 4; ++j) {
         fmpi[j] = 0;
         for (int k = 1; k < 4; ++k) {
            fmpi[j] += a[k - 1][j - 1] * cmpi[k];
         }
      }
      for (int j = 4; j < 10; ++j) {
         fmpi[j] = 0;
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

void cmpToFmp_cu(PMEUnit pme_u, const real (*cmp)[10], real (*fmp)[10])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   launch_k1s(g::s0, n, cmpToFmp_cu1, n, cmp, fmp, nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
}

__global__
void cuindToFuind_cu1(int n, const real (*restrict cind)[3], const real (*restrict cinp)[3], real (*restrict fuind)[3],
   real (*restrict fuinp)[3], int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
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

   for (int i = ITHREAD; i < n; i += STRIDE) {
      for (int j = 0; j < 3; ++j) {
         fuind[i][j] = a[0][j] * cind[i][0] + a[1][j] * cind[i][1] + a[2][j] * cind[i][2];
         fuinp[i][j] = a[0][j] * cinp[i][0] + a[1][j] * cinp[i][1] + a[2][j] * cinp[i][2];
      }
   }
}

void cuindToFuind_cu(PMEUnit pme_u, const real (*cind)[3], const real (*cinp)[3], //
   real (*fuind)[3], real (*fuinp)[3])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   launch_k1s(g::s0, n, cuindToFuind_cu1, n, cind, cinp, fuind, fuinp, nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
}

__global__
void fphiToCphi_cu1(int n, const real (*restrict fphi)[20], real (*restrict cphi)[10], int nfft1, int nfft2, int nfft3,
   TINKER_IMAGE_PARAMS)
{
   // see also subroutine frac_to_cart in pmestuf.f
   // set the reciprocal vector transformation matrix
   real a[3][3];
   a[0][0] = nfft1 * recipa.x;
   a[1][0] = nfft2 * recipb.x;
   a[2][0] = nfft3 * recipc.x;
   a[0][1] = nfft1 * recipa.y;
   a[1][1] = nfft2 * recipb.y;
   a[2][1] = nfft3 * recipc.y;
   a[0][2] = nfft1 * recipa.z;
   a[1][2] = nfft2 * recipb.z;
   a[2][2] = nfft3 * recipc.z;

   for (int iatom = ITHREAD; iatom < n; iatom += STRIDE) {
      // data qi1  / 1, 2, 3, 1, 1, 2 /
      // data qi2  / 1, 2, 3, 2, 3, 3 /
      constexpr int qi1[] = {0, 1, 2, 0, 0, 1};
      constexpr int qi2[] = {0, 1, 2, 1, 2, 2};

      real ftc6[6][6];
      // get the fractional to Cartesian conversion matrix
      for (int i1 = 0; i1 < 3; ++i1) {
         int k = qi1[i1];
         for (int i2 = 0; i2 < 3; ++i2) {
            int i = qi1[i2];
            ftc6[i2][i1] = a[i][k] * a[i][k];
         }
         for (int i2 = 3; i2 < 6; ++i2) {
            int i = qi1[i2];
            int j = qi2[i2];
            ftc6[i2][i1] = 2 * a[i][k] * a[j][k];
         }
      }

      for (int i1 = 3; i1 < 6; ++i1) {
         int k = qi1[i1];
         int m = qi2[i1];
         for (int i2 = 0; i2 < 3; ++i2) {
            int i = qi1[i2];
            ftc6[i2][i1] = a[i][k] * a[i][m];
         }
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
      for (int j = 1; j < 4; ++j) {
         cphii[j] = 0;
         for (int k = 1; k < 4; ++k) {
            cphii[j] += a[k - 1][j - 1] * fphii[k];
         }
      }
      for (int j = 4; j < 10; ++j) {
         cphii[j] = 0;
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

void fphiToCphi_cu(PMEUnit pme_u, const real (*fphi)[20], real (*cphi)[10])
{
   auto& st = *pme_u;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   launch_k1s(g::s0, n, fphiToCphi_cu1, n, fphi, cphi, nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
}
}
