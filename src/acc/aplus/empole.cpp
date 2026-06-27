#include "ff/atom.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/switch.h"
#include "seq/pairmpoleaplus.h"
#include "tool/gpucard.h"

namespace tinker {
#define DEVICE_PTRS x, y, z, demx, demy, demz, rpole, pcore, pval, palpha, pot, nem, em, vir_em, trqx, trqy, trqz
template <class Ver, class ETYP, bool CFLX>
static void empoleAplus_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const real f = electric / dielec;

   real aewald = 0;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switchOff(Switch::EWALD);
      const PMEUnit pu = epme_unit;
      aewald = pu->aewald;
   } else {
      off = switchOff(Switch::MPOLE);
   }

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();
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
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;
      MAYBE_UNUSED real poti = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi,poti)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real corek = pcore[k];
         real alphak = palpha[k];
         real valk = pval[k];

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            MAYBE_UNUSED real e;
            MAYBE_UNUSED real pota, potb;
            pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, 1, //
               ci, dix, diy, diz, corei, vali, alphai,                         //
               qixx, qixy, qixz, qiyy, qiyz, qizz,                             //
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], corek, valk, alphak,
               rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY],
               rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], f, aewald, e, pota, potb, pgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
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

               // Charge flux term
               if CONSTEXPR (CFLX) {
                  poti += pota;
                  atomic_add(potb, pot, k);
               }
            } // end if (do_g)
         }    // end if (r2 <= off2)
      }       // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, demx, i);
         atomic_add(gyi, demy, i);
         atomic_add(gzi, demz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (CFLX) {
            atomic_add(poti, pot, i);
         }
      }

      if CONSTEXPR (eq<ETYP, EWALD>()) {
         const real aewald_sq_2 = 2 * aewald * aewald;
         const real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);
         real cii = ci * ci;
         real dii = dix * dix + diy * diy + diz * diz;
         real qii = 2 * (qixy * qixy + qixz * qixz + qiyz * qiyz) + qixx * qixx + qiyy * qiyy + qizz * qizz;

         if CONSTEXPR (do_e) {
            int offset = i & (bufsize - 1);
            real e = fterm * (cii + aewald_sq_2 * (dii / 3 + 2 * aewald_sq_2 * qii * (real)0.2));
            atomic_add(e, em, offset);
            if CONSTEXPR (do_a)
               atomic_add(1, nem, offset);
         } // end if (do_e)

         if CONSTEXPR (CFLX) {
            real cfl_term = 2 * fterm * ci;
            atomic_add(cfl_term, pot, i);
         }
      }
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mdwexclude,mdwexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real mscale = mdwexclude_scale[ii][0] - 1;

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

      real corek = pcore[k];
      real alphak = palpha[k];
      real valk = pval[k];

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2 and mscale != 0) {

         MAYBE_UNUSED real e;
         MAYBE_UNUSED real pota, potb;

         // Compute scaled interaction
         pair_mpole_chgpen_aplus<do_e, do_g, NON_EWALD, CFLX>(r2, xr, yr, zr, mscale, //
            ci, dix, diy, diz, corei, vali, alphai,                                   //
            qixx, qixy, qixz, qiyy, qiyz, qizz,                                       //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], corek, valk, alphak,
            rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY],
            rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], f, 0, e, pota, potb, pgrad);

         if CONSTEXPR (do_a)
            if (mscale == -1 and e != 0)
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

            if CONSTEXPR (CFLX) {
               atomic_add(pota, pot, i);
               atomic_add(potb, pot, k);
            }
         } // end if (do_g)
      }    // end if (r2 <= off2)
   }       // end for (int ii)
}

void empoleAplusNonEwald_acc(int vers, int use_cf)
{
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleAplus_acc1<calc::V0, NON_EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleAplus_acc1<calc::V1, NON_EWALD, 1>();
      } else if (vers == calc::v3) {
         // empoleAplus_acc1<calc::V3, NON_EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleAplus_acc1<calc::V4, NON_EWALD, 1>();
      } else if (vers == calc::v5) {
         empoleAplus_acc1<calc::V5, NON_EWALD, 1>();
      } else if (vers == calc::v6) {
         empoleAplus_acc1<calc::V6, NON_EWALD, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleAplus_acc1<calc::V0, NON_EWALD, 0>();
      } else if (vers == calc::v1) {
         empoleAplus_acc1<calc::V1, NON_EWALD, 0>();
      } else if (vers == calc::v3) {
         empoleAplus_acc1<calc::V3, NON_EWALD, 0>();
      } else if (vers == calc::v4) {
         empoleAplus_acc1<calc::V4, NON_EWALD, 0>();
      } else if (vers == calc::v5) {
         empoleAplus_acc1<calc::V5, NON_EWALD, 0>();
      } else if (vers == calc::v6) {
         empoleAplus_acc1<calc::V6, NON_EWALD, 0>();
      }
   }
}

void empoleAplusEwaldRealSelf_acc(int vers, int use_cf)
{
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleAplus_acc1<calc::V0, EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleAplus_acc1<calc::V1, EWALD, 1>();
      } else if (vers == calc::v3) {
         // empoleAplus_acc1<calc::V3, EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleAplus_acc1<calc::V4, EWALD, 1>();
      } else if (vers == calc::v5) {
         empoleAplus_acc1<calc::V5, EWALD, 1>();
      } else if (vers == calc::v6) {
         empoleAplus_acc1<calc::V6, EWALD, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleAplus_acc1<calc::V0, EWALD, 0>();
      } else if (vers == calc::v1) {
         empoleAplus_acc1<calc::V1, EWALD, 0>();
      } else if (vers == calc::v3) {
         empoleAplus_acc1<calc::V3, EWALD, 0>();
      } else if (vers == calc::v4) {
         empoleAplus_acc1<calc::V4, EWALD, 0>();
      } else if (vers == calc::v5) {
         empoleAplus_acc1<calc::V5, EWALD, 0>();
      } else if (vers == calc::v6) {
         empoleAplus_acc1<calc::V6, EWALD, 0>();
      }
   }
}
}
