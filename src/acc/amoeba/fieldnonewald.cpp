#include "ff/atom.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/pair_field.h"
#include "tool/gpucard.h"

#define TINKER9_POLPAIR 2

namespace tinker {
// see also subroutine dfield0b in induce.f
#define DFIELD_DPTRS x, y, z, thole, pdamp, field, fieldp, rpole, jpolar, thlval
void dfieldNonEwald_acc(real (*field)[3], real (*fieldp)[3])
{
   darray::zero(g::q0, n, field, fieldp);

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DFIELD_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][MPL_PME_0];
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];
      real pdi = pdamp[i];
#if TINKER9_POLPAIR == 2
      int jpi = jpolar[i];
#else
      real pti = thole[i];
#endif
      real gxi = 0, gyi = 0, gzi = 0;
      real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fip = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            real3 fkp = make_real3(0, 0, 0);
#if TINKER9_POLPAIR == 2
            int jpk = jpolar[k];
            real pga = thlval[njpolar * jpi + jpk];
            pair_dfield<NON_EWALD>(  //
               r2, xr, yr, zr, 1, 1, //
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi,
               pga, //
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
               rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
               rpole[k][MPL_PME_ZZ], pdamp[k],
               pga, //
               0, fid, fip, fkd, fkp);
#else
            pair_dfield<NON_EWALD>(  //
               r2, xr, yr, zr, 1, 1, //
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi,
               pti, //
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
               rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
               rpole[k][MPL_PME_ZZ], pdamp[k],
               thole[k], //
               0, fid, fip, fkd, fkp);
#endif

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;
            txi += fip.x;
            tyi += fip.y;
            tzi += fip.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
            atomic_add(fkp.x, &fieldp[k][0]);
            atomic_add(fkp.y, &fieldp[k][1]);
            atomic_add(fkp.z, &fieldp[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
      atomic_add(txi, &fieldp[i][0]);
      atomic_add(tyi, &fieldp[i][1]);
      atomic_add(tzi, &fieldp[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DFIELD_DPTRS,dpexclude,dpexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < ndpexclude; ++ii) {
      int i = dpexclude[ii][0];
      int k = dpexclude[ii][1];
      real dscale = dpexclude_scale[ii][0] - 1;
      real pscale = dpexclude_scale[ii][1] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][MPL_PME_0];
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];
      real pdi = pdamp[i];
#if TINKER9_POLPAIR == 2
      int jpi = jpolar[i];
#else
      real pti = thole[i];
#endif

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fip = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         real3 fkp = make_real3(0, 0, 0);
#if TINKER9_POLPAIR == 2
         int jpk = jpolar[k];
         real pga = thlval[njpolar * jpi + jpk];
         pair_dfield<NON_EWALD>(                                             //
            r2, xr, yr, zr, dscale, pscale,                                  //
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi, pga, //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
            rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
            rpole[k][MPL_PME_ZZ], pdamp[k], pga, //
            0, fid, fip, fkd, fkp);
#else
         pair_dfield<NON_EWALD>(                                             //
            r2, xr, yr, zr, dscale, pscale,                                  //
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi, pti, //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
            rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
            rpole[k][MPL_PME_ZZ], pdamp[k], thole[k], //
            0, fid, fip, fkd, fkp);
#endif

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);
         atomic_add(fip.x, &fieldp[i][0]);
         atomic_add(fip.y, &fieldp[i][1]);
         atomic_add(fip.z, &fieldp[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
         atomic_add(fkp.x, &fieldp[k][0]);
         atomic_add(fkp.y, &fieldp[k][1]);
         atomic_add(fkp.z, &fieldp[k][2]);
      }
   }
}

// see also subroutine ufield0b in induce.f
#define UFIELD_DPTRS x, y, z, thole, pdamp, field, fieldp, uind, uinp, jpolar, thlval
void ufieldNonEwald_acc(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   darray::zero(g::q0, n, field, fieldp);

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(UFIELD_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uindi0 = uind[i][0];
      real uindi1 = uind[i][1];
      real uindi2 = uind[i][2];
      real uinpi0 = uinp[i][0];
      real uinpi1 = uinp[i][1];
      real uinpi2 = uinp[i][2];
      real pdi = pdamp[i];
#if TINKER9_POLPAIR == 2
      int jpi = jpolar[i];
#else
      real pti = thole[i];
#endif
      real gxi = 0, gyi = 0, gzi = 0;
      real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fip = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            real3 fkp = make_real3(0, 0, 0);
#if TINKER9_POLPAIR == 2
            int jpk = jpolar[k];
            real pga = thlval[njpolar * jpi + jpk];
            pair_ufield<NON_EWALD>(                                      //
               r2, xr, yr, zr, 1,                                        //
               uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pga, //
               uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
               pga, //
               0, fid, fip, fkd, fkp);
#else
            pair_ufield<NON_EWALD>(                                      //
               r2, xr, yr, zr, 1,                                        //
               uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pti, //
               uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
               thole[k], //
               0, fid, fip, fkd, fkp);
#endif

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;
            txi += fip.x;
            tyi += fip.y;
            tzi += fip.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
            atomic_add(fkp.x, &fieldp[k][0]);
            atomic_add(fkp.y, &fieldp[k][1]);
            atomic_add(fkp.z, &fieldp[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
      atomic_add(txi, &fieldp[i][0]);
      atomic_add(tyi, &fieldp[i][1]);
      atomic_add(tzi, &fieldp[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(UFIELD_DPTRS,uexclude,uexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nuexclude; ++ii) {
      int i = uexclude[ii][0];
      int k = uexclude[ii][1];
      real uscale = uexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uindi0 = uind[i][0];
      real uindi1 = uind[i][1];
      real uindi2 = uind[i][2];
      real uinpi0 = uinp[i][0];
      real uinpi1 = uinp[i][1];
      real uinpi2 = uinp[i][2];
      real pdi = pdamp[i];
#if TINKER9_POLPAIR == 2
      int jpi = jpolar[i];
#else
      real pti = thole[i];
#endif

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fip = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         real3 fkp = make_real3(0, 0, 0);
#if TINKER9_POLPAIR == 2
         int jpk = jpolar[k];
         real pga = thlval[njpolar * jpi + jpk];
         pair_ufield<NON_EWALD>(                                      //
            r2, xr, yr, zr, uscale,                                   //
            uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pga, //
            uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
            pga, //
            0, fid, fip, fkd, fkp);
#else
         pair_ufield<NON_EWALD>(                                      //
            r2, xr, yr, zr, uscale,                                   //
            uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pti, //
            uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
            thole[k], //
            0, fid, fip, fkd, fkp);
#endif

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);
         atomic_add(fip.x, &fieldp[i][0]);
         atomic_add(fip.y, &fieldp[i][1]);
         atomic_add(fip.z, &fieldp[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
         atomic_add(fkp.x, &fieldp[k][0]);
         atomic_add(fkp.y, &fieldp[k][1]);
         atomic_add(fkp.z, &fieldp[k][2]);
      }
   }
}
}
