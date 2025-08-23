#include "ff/energy.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/pair_mpole.h"
#include "tool/gpucard.h"
#include <tinker/detail/extfld.hh>

namespace tinker {
#define DEVICE_PTRS x, y, z, demx, demy, demz, rpole, nem, em, vir_em, trqx, trqy, trqz
template <class Ver>
void empoleNonEwald_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const real f = electric / dielec;

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   auto bufsize = bufferSize();
   PairMPoleGrad pgrad;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mlst)
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
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            MAYBE_UNUSED real e;
            pair_mpole<do_e, do_g, NON_EWALD>(                        //
               r2, xr, yr, zr, 1,                                     //
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, //
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
               rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
               rpole[k][MPL_PME_ZZ], //
               f, 0, e, pgrad);

            if CONSTEXPR (do_a)
               atomic_add(1, nem, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, em, offset);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, demx, k);
               atomic_add(-pgrad.frcy, demy, k);
               atomic_add(-pgrad.frcz, demz, k);

               txi += pgrad.ttmi[0];
               tyi += pgrad.ttmi[1];
               tzi += pgrad.ttmi[2];
               atomic_add(pgrad.ttmk[0], trqx, k);
               atomic_add(pgrad.ttmk[1], trqy, k);
               atomic_add(pgrad.ttmk[2], trqz, k);

               // virial

               if CONSTEXPR (do_v) {
                  real vxx = -xr * pgrad.frcx;
                  real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
                  real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
                  real vyy = -yr * pgrad.frcy;
                  real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
                  real vzz = -zr * pgrad.frcz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
               } // end if (do_v)
            }    // end if (do_g)
         }       // end if (r2 <= off2)
      }          // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, demx, i);
         atomic_add(gyi, demy, i);
         atomic_add(gzi, demz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
      }
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mexclude,mexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < nmexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mexclude[ii][0];
      int k = mexclude[ii][1];
      real mscale = mexclude_scale[ii] - 1;

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

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         MAYBE_UNUSED real e;
         pair_mpole<do_e, do_g, NON_EWALD>(                        //
            r2, xr, yr, zr, mscale,                                //
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX],
            rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
            rpole[k][MPL_PME_ZZ], //
            f, 0, e, pgrad);

         if CONSTEXPR (do_a)
            if (mscale == -1)
               atomic_add(-1, nem, offset);
         if CONSTEXPR (do_e)
            atomic_add(e, em, offset);
         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, demx, i);
            atomic_add(pgrad.frcy, demy, i);
            atomic_add(pgrad.frcz, demz, i);
            atomic_add(-pgrad.frcx, demx, k);
            atomic_add(-pgrad.frcy, demy, k);
            atomic_add(-pgrad.frcz, demz, k);

            atomic_add(pgrad.ttmi[0], trqx, i);
            atomic_add(pgrad.ttmi[1], trqy, i);
            atomic_add(pgrad.ttmi[2], trqz, i);
            atomic_add(pgrad.ttmk[0], trqx, k);
            atomic_add(pgrad.ttmk[1], trqy, k);
            atomic_add(pgrad.ttmk[2], trqz, k);

            // virial

            if CONSTEXPR (do_v) {
               real vxx = -xr * pgrad.frcx;
               real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               real vyy = -yr * pgrad.frcy;
               real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               real vzz = -zr * pgrad.frcz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
            } // end if (do_v)
         }    // end if (do_g)
      }
   }
}

void empoleNonEwald_acc(int vers)
{
   if (vers == calc::v0)
      empoleNonEwald_acc1<calc::V0>();
   else if (vers == calc::v1)
      empoleNonEwald_acc1<calc::V1>();
   else if (vers == calc::v3)
      empoleNonEwald_acc1<calc::V3>();
   else if (vers == calc::v4)
      empoleNonEwald_acc1<calc::V4>();
   else if (vers == calc::v5)
      empoleNonEwald_acc1<calc::V5>();
   else if (vers == calc::v6)
      empoleNonEwald_acc1<calc::V6>();
}

void exfieldDipole_acc(int vers)
{
   bool do_e = vers & calc::energy;
   bool do_a = vers & calc::analyz;
   bool do_g = vers & calc::grad;
   bool do_v = vers & calc::virial;

   auto bufsize = bufferSize();
   real f = electric / dielec;
   real ef1 = extfld::exfld[0], ef2 = extfld::exfld[1], ef3 = extfld::exfld[2];

   #pragma acc parallel async deviceptr(rpole,nem,em,vir_em,x,y,z,demx,demy,demz,trqx,trqy,trqz)
   #pragma acc loop independent
   for (int ii = 0; ii < n; ++ii) {
      int offset = ii & (bufsize - 1);
      real xi = x[ii], yi = y[ii], zi = z[ii];
      real ci = rpole[ii][0], dix = rpole[ii][1], diy = rpole[ii][2], diz = rpole[ii][3];

      if (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * (ci * phi + dix * ef1 + diy * ef2 + diz * ef3);
         atomic_add(e, em, offset);
         if (do_a)
            atomic_add(1, nem, offset);
      }
      if (do_g) {
         // torque due to the dipole
         real tx = f * (diy * ef3 - diz * ef2);
         real ty = f * (diz * ef1 - dix * ef3);
         real tz = f * (dix * ef2 - diy * ef1);
         atomic_add(tx, trqx, ii);
         atomic_add(ty, trqy, ii);
         atomic_add(tz, trqz, ii);
         // gradient and virial due to the monopole
         real frx = -f * ef1 * ci;
         real fry = -f * ef2 * ci;
         real frz = -f * ef3 * ci;
         atomic_add(frx, demx, ii);
         atomic_add(fry, demy, ii);
         atomic_add(frz, demz, ii);
         if (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
         }
      }
   }
}

void extfieldModifyDField_acc(real (*field)[3], real (*fieldp)[3])
{
   real ex1 = extfld::exfld[0];
   real ex2 = extfld::exfld[1];
   real ex3 = extfld::exfld[2];

   if (fieldp) {
      #pragma acc parallel loop independent async deviceptr(field,fieldp)
      for (int i = 0; i < n; ++i) {
         field[i][0] += ex1;
         field[i][1] += ex2;
         field[i][2] += ex3;
         fieldp[i][0] += ex1;
         fieldp[i][1] += ex2;
         fieldp[i][2] += ex3;
      }
   } else {
      #pragma acc parallel loop independent async deviceptr(field)
      for (int i = 0; i < n; ++i) {
         field[i][0] += ex1;
         field[i][1] += ex2;
         field[i][2] += ex3;
      }
   }
}
}
