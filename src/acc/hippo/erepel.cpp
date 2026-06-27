#include "ff/atom.h"
#include "ff/hippo/erepel.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/pair_repel.h"
#include "tool/gpucard.h"

namespace tinker {
#define DEVICE_PTRS x, y, z, derx, dery, derz, rrepole, sizpr, dmppr, elepr, nrep, er, vir_er, trqx, trqy, trqz, mut
template <class Ver>
static void erepel_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();

   PairRepelGrad pgrad;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rrepole[i][MPL_PME_0];
      real dix = rrepole[i][MPL_PME_X];
      real diy = rrepole[i][MPL_PME_Y];
      real diz = rrepole[i][MPL_PME_Z];
      real qixx = rrepole[i][MPL_PME_XX];
      real qixy = rrepole[i][MPL_PME_XY];
      real qixz = rrepole[i][MPL_PME_XZ];
      real qiyy = rrepole[i][MPL_PME_YY];
      real qiyz = rrepole[i][MPL_PME_YZ];
      real qizz = rrepole[i][MPL_PME_ZZ];
      real sizi = sizpr[i];
      real dmpi = dmppr[i];
      real vali = elepr[i];
      int imut = mut[i];

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
         real sizk = sizpr[k];
         real dmpk = dmppr[k];
         real valk = elepr[k];
         int kmut = mut[k];

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real e;
            zero(pgrad);
            real vlambda = pair_vlambda(vlam, vcouple, imut, kmut);
            pair_repel<do_g, 1>( //
               r2, 1, vlambda, cut, off, xr, yr, zr, sizi, dmpi, vali, ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz,
               qizz, sizk, dmpk, valk, rrepole[k][MPL_PME_0], rrepole[k][MPL_PME_X], rrepole[k][MPL_PME_Y],
               rrepole[k][MPL_PME_Z], rrepole[k][MPL_PME_XX], rrepole[k][MPL_PME_XY], rrepole[k][MPL_PME_XZ],
               rrepole[k][MPL_PME_YY], rrepole[k][MPL_PME_YZ], rrepole[k][MPL_PME_ZZ], e, pgrad);
            if CONSTEXPR (do_a)
               if (e != 0)
                  atomic_add(1, nrep, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, er, offset);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, derx, k);
               atomic_add(-pgrad.frcy, dery, k);
               atomic_add(-pgrad.frcz, derz, k);

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               atomic_add(pgrad.ttqk[0], trqx, k);
               atomic_add(pgrad.ttqk[1], trqy, k);
               atomic_add(pgrad.ttqk[2], trqz, k);

               // virial

               if CONSTEXPR (do_v) {
                  real vxx = -xr * pgrad.frcx;
                  real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
                  real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
                  real vyy = -yr * pgrad.frcy;
                  real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
                  real vzz = -zr * pgrad.frcz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_er, offset);
               } // end if (do_v)
            }    // end if (do_g)
         }       // end if (r2 <= off2)
      }          // end for (int kk)
      if CONSTEXPR (do_g) {
         atomic_add(gxi, derx, i);
         atomic_add(gyi, dery, i);
         atomic_add(gzi, derz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
      }
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,repexclude,repexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < nrepexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = repexclude[ii][0];
      int k = repexclude[ii][1];
      real rscale = repexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rrepole[i][MPL_PME_0];
      real dix = rrepole[i][MPL_PME_X];
      real diy = rrepole[i][MPL_PME_Y];
      real diz = rrepole[i][MPL_PME_Z];
      real qixx = rrepole[i][MPL_PME_XX];
      real qixy = rrepole[i][MPL_PME_XY];
      real qixz = rrepole[i][MPL_PME_XZ];
      real qiyy = rrepole[i][MPL_PME_YY];
      real qiyz = rrepole[i][MPL_PME_YZ];
      real qizz = rrepole[i][MPL_PME_ZZ];
      real sizi = sizpr[i];
      real dmpi = dmppr[i];
      real vali = elepr[i];
      int imut = mut[i];
      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;
      real sizk = sizpr[k];
      real dmpk = dmppr[k];
      real valk = elepr[k];
      int kmut = mut[k];

      zero(pgrad);
      real r2 = image2(xr, yr, zr);
      if (r2 <= off2 and rscale != 0) {
         real e;
         real vlambda = pair_vlambda(vlam, vcouple, imut, kmut);
         pair_repel<do_g, 1>( //
            r2, rscale, vlambda, cut, off, xr, yr, zr, sizi, dmpi, vali, ci, dix, diy, diz, qixx, qixy, qixz, qiyy,
            qiyz, qizz, sizk, dmpk, valk, rrepole[k][MPL_PME_0], rrepole[k][MPL_PME_X], rrepole[k][MPL_PME_Y],
            rrepole[k][MPL_PME_Z], rrepole[k][MPL_PME_XX], rrepole[k][MPL_PME_XY], rrepole[k][MPL_PME_XZ],
            rrepole[k][MPL_PME_YY], rrepole[k][MPL_PME_YZ], rrepole[k][MPL_PME_ZZ], e, pgrad);
         if CONSTEXPR (do_a)
            if (rscale == -1 and e != 0)
               atomic_add(-1, nrep, offset);
         if CONSTEXPR (do_e)
            atomic_add(e, er, offset);
         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, derx, i);
            atomic_add(pgrad.frcy, dery, i);
            atomic_add(pgrad.frcz, derz, i);
            atomic_add(-pgrad.frcx, derx, k);
            atomic_add(-pgrad.frcy, dery, k);
            atomic_add(-pgrad.frcz, derz, k);

            atomic_add(pgrad.ttqi[0], trqx, i);
            atomic_add(pgrad.ttqi[1], trqy, i);
            atomic_add(pgrad.ttqi[2], trqz, i);
            atomic_add(pgrad.ttqk[0], trqx, k);
            atomic_add(pgrad.ttqk[1], trqy, k);
            atomic_add(pgrad.ttqk[2], trqz, k);

            // virial

            if CONSTEXPR (do_v) {
               real vxx = -xr * pgrad.frcx;
               real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               real vyy = -yr * pgrad.frcy;
               real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               real vzz = -zr * pgrad.frcz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_er, offset);
            } // end if (do_v)
         }    // end if (do_g)
      }       // end if (r2 <= off2)
   }          // end for (int i)
}

void erepel_acc(int vers)
{
   if (vers == calc::v0)
      erepel_acc1<calc::V0>();
   else if (vers == calc::v1)
      erepel_acc1<calc::V1>();
   else if (vers == calc::v3)
      erepel_acc1<calc::V3>();
   else if (vers == calc::v4)
      erepel_acc1<calc::V4>();
   else if (vers == calc::v5)
      erepel_acc1<calc::V5>();
   else if (vers == calc::v6)
      erepel_acc1<calc::V6>();
}
}
