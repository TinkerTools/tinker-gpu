#include "ff/atom.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/potent.h"
#include "ff/switch.h"
#include "seq/pair_field_chgpen.h"
#include "tool/gpucard.h"

namespace tinker {
// see also subroutine udirect2b / dfield_chgpen0c in induce.f
#define DFIELD_DPTRS x, y, z, pcore, pval, palpha, field, rpole
template <class ETYP>
static void dfieldChgpen_acc1(real (*field)[3])
{
   real aewald = 0;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switchOff(Switch::EWALD);
      const PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   } else {
      darray::zero(g::q0, n, field);
      off = switchOff(Switch::MPOLE);
   }

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
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];
      real gxi = 0, gyi = 0, gzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;

      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            pair_dfield_chgpen<ETYP>( //
               r2, xr, yr, zr, 1, ci, dix, diy, diz, corei, vali, alphai, qixx, qixy, qixz, qiyy, qiyz, qizz,
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], pcore[k], pval[k],
               palpha[k], rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY],
               rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], aewald, fid.x, fid.y, fid.z, fkd.x, fkd.y, fkd.z);

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DFIELD_DPTRS,mdwexclude,mdwexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real dscale = mdwexclude_scale[ii][1] - 1;

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
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         pair_dfield_chgpen<NON_EWALD>(r2, xr, yr, zr, dscale, ci, dix, diy, diz, corei, vali, alphai, qixx, qixy, qixz,
            qiyy, qiyz, qizz, rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z],
            pcore[k], pval[k], palpha[k], rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ],
            rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], 0, fid.x, fid.y, fid.z, fkd.x, fkd.y,
            fkd.z);

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
      }
   }
}

void dfieldChgpenNonEwald_acc(real (*field)[3])
{
   dfieldChgpen_acc1<NON_EWALD>(field);
}

void dfieldChgpenEwaldReal_acc(real (*field)[3])
{
   dfieldChgpen_acc1<EWALD>(field);
}

#define UFIELD_DPTRS x, y, z, pcore, pval, palpha, field, uind
template <class ETYP>
static void ufieldChgpen_acc1(const real (*uind)[3], real (*field)[3])
{
   real aewald = 0;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switchOff(Switch::EWALD);
      const PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   } else {
      darray::zero(g::q0, n, field);
      off = switchOff(Switch::MPOLE);
   }

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
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];
      real gxi = 0, gyi = 0, gzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            pair_ufield_chgpen<ETYP>(r2, xr, yr, zr, 1, uindi0, uindi1, uindi2, corei, vali, alphai, uind[k][0],
               uind[k][1], uind[k][2], pcore[k], pval[k], palpha[k], aewald, fid.x, fid.y, fid.z, fkd.x, fkd.y, fkd.z);

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(UFIELD_DPTRS,mdwexclude,mdwexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real wscale = mdwexclude_scale[ii][2] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uindi0 = uind[i][0];
      real uindi1 = uind[i][1];
      real uindi2 = uind[i][2];
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         pair_ufield_chgpen<NON_EWALD>(r2, xr, yr, zr, wscale, uindi0, uindi1, uindi2, corei, vali, alphai, uind[k][0],
            uind[k][1], uind[k][2], pcore[k], pval[k], palpha[k], 0, fid.x, fid.y, fid.z, fkd.x, fkd.y, fkd.z);

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
      }
   }
}

void ufieldChgpenNonEwald_acc(const real (*uind)[3], real (*field)[3])
{
   ufieldChgpen_acc1<NON_EWALD>(uind, field);
}

void ufieldChgpenEwaldReal_acc(const real (*uind)[3], real (*field)[3])
{
   ufieldChgpen_acc1<EWALD>(uind, field);
}
}
