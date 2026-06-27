#include "ff/atom.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/switch.h"
#include "seq/pairpolaraplus.h"
#include "tool/gpucard.h"

namespace tinker {
#define POLAR_DPTRS x, y, z, depx, depy, depz, rpole, thole, dirdamp, pdamp, pot, uind, nep, ep, vir_ep, ufld, dufld
template <class Ver, class ETYP, bool CFLX>
static void epolarAplus_acc1(const real (*uind)[3])
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, ufld, dufld);

   real aewald = 0;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switchOff(Switch::EWALD);
      const PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   } else {
      off = switchOff(Switch::MPOLE);
   }

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();
   PairPolarGrad pgrad;

   const real f = 0.5f * electric / dielec;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(POLAR_DPTRS,mlst)
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
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];
      real pdi = pdamp[i];
      real pti = thole[i];
      real ddi = dirdamp[i];

      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;
      MAYBE_UNUSED real du0 = 0, du1 = 0, du2 = 0, du3 = 0, du4 = 0, du5 = 0;
      MAYBE_UNUSED real poti = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi,poti,\
                  du0,du1,du2,du3,du4,du5)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = kk & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         zero(pgrad);
         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            MAYBE_UNUSED real e;
            MAYBE_UNUSED real pota, potb;

            real ck = rpole[k][MPL_PME_0];
            real dkx = rpole[k][MPL_PME_X];
            real dky = rpole[k][MPL_PME_Y];
            real dkz = rpole[k][MPL_PME_Z];
            real qkxx = rpole[k][MPL_PME_XX];
            real qkxy = rpole[k][MPL_PME_XY];
            real qkxz = rpole[k][MPL_PME_XZ];
            real qkyy = rpole[k][MPL_PME_YY];
            real qkyz = rpole[k][MPL_PME_YZ];
            real qkzz = rpole[k][MPL_PME_ZZ];
            real ukx = uind[k][0];
            real uky = uind[k][1];
            real ukz = uind[k][2];
            real ptk = thole[k];
            real pdk = pdamp[k];
            real ddk = dirdamp[k];

            pair_polar_aplus<do_e, do_g, ETYP, CFLX>( //
               r2, xr, yr, zr, 1, 1, ci, dix, diy, diz, pdi, pti, ddi, qixx, qixy, qixz, qiyy, qiyz, qizz, uix, uiy,
               uiz, ck, dkx, dky, dkz, pdk, ptk, ddk, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukx, uky, ukz, f, aewald, e,
               pota, potb, pgrad);

            if CONSTEXPR (do_a)
               atomic_add(1, nep, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, ep, offset);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, depx, k);
               atomic_add(-pgrad.frcy, depy, k);
               atomic_add(-pgrad.frcz, depz, k);

               txi += pgrad.ufldi[0];
               tyi += pgrad.ufldi[1];
               tzi += pgrad.ufldi[2];
               atomic_add(pgrad.ufldk[0], &ufld[k][0]);
               atomic_add(pgrad.ufldk[1], &ufld[k][1]);
               atomic_add(pgrad.ufldk[2], &ufld[k][2]);

               du0 += pgrad.dufldi[0];
               du1 += pgrad.dufldi[1];
               du2 += pgrad.dufldi[2];
               du3 += pgrad.dufldi[3];
               du4 += pgrad.dufldi[4];
               du5 += pgrad.dufldi[5];
               atomic_add(pgrad.dufldk[0], &dufld[k][0]);
               atomic_add(pgrad.dufldk[1], &dufld[k][1]);
               atomic_add(pgrad.dufldk[2], &dufld[k][2]);
               atomic_add(pgrad.dufldk[3], &dufld[k][3]);
               atomic_add(pgrad.dufldk[4], &dufld[k][4]);
               atomic_add(pgrad.dufldk[5], &dufld[k][5]);

               if CONSTEXPR (do_v) {
                  real vxx = -xr * pgrad.frcx;
                  real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
                  real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
                  real vyy = -yr * pgrad.frcy;
                  real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
                  real vzz = -zr * pgrad.frcz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
               } // end if (do_v)
               // Charge flux term
               if CONSTEXPR (CFLX) {
                  poti += pota;
                  atomic_add(potb, pot, k);
               } // end CFLX
            }    // end if (r2 <= off2)
         }       // end if (do_g)
      }          // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, depx, i);
         atomic_add(gyi, depy, i);
         atomic_add(gzi, depz, i);
         atomic_add(txi, &ufld[i][0]);
         atomic_add(tyi, &ufld[i][1]);
         atomic_add(tzi, &ufld[i][2]);
         atomic_add(du0, &dufld[i][0]);
         atomic_add(du1, &dufld[i][1]);
         atomic_add(du2, &dufld[i][2]);
         atomic_add(du3, &dufld[i][3]);
         atomic_add(du4, &dufld[i][4]);
         atomic_add(du5, &dufld[i][5]);
         if CONSTEXPR (CFLX) {
            atomic_add(poti, pot, i);
         }
      }
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(POLAR_DPTRS,dpuexclude,dpuexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < ndpuexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = dpuexclude[ii][0];
      int k = dpuexclude[ii][1];
      real pscale = dpuexclude_scale[ii][1] - 1; // AMOEBA Plus d-equals-p; use p
      real uscale = dpuexclude_scale[ii][2] - 1;

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
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];
      real pti = thole[i];
      real pdi = pdamp[i];
      real ddi = dirdamp[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real ptk = thole[k];
      real pdk = pdamp[k];
      real ddk = dirdamp[k];

      bool incl = pscale != 0 or uscale != 0;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2 and incl) {

         MAYBE_UNUSED real e;
         MAYBE_UNUSED real pota, potb;

         pair_polar_aplus<do_e, do_g, NON_EWALD, CFLX>(        //
            r2, xr, yr, zr, pscale, uscale,                    //
            ci, dix, diy, diz, pdi, pti, ddi,                  //
            qixx, qixy, qixz, qiyy, qiyz, qizz, uix, uiy, uiz, //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z], pdk, ptk, ddk,
            rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY],
            rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], uind[k][0], uind[k][1], uind[k][2], f, 0, e, pota, potb, pgrad);

         if CONSTEXPR (do_a)
            if (pscale == -1 and e != 0)
               atomic_add(-1, nep, offset);

         if CONSTEXPR (do_e)
            atomic_add(e, ep, offset);
         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, depx, i);
            atomic_add(pgrad.frcy, depy, i);
            atomic_add(pgrad.frcz, depz, i);
            atomic_add(-pgrad.frcx, depx, k);
            atomic_add(-pgrad.frcy, depy, k);
            atomic_add(-pgrad.frcz, depz, k);

            atomic_add(pgrad.ufldi[0], &ufld[i][0]);
            atomic_add(pgrad.ufldi[1], &ufld[i][1]);
            atomic_add(pgrad.ufldi[2], &ufld[i][2]);
            atomic_add(pgrad.ufldk[0], &ufld[k][0]);
            atomic_add(pgrad.ufldk[1], &ufld[k][1]);
            atomic_add(pgrad.ufldk[2], &ufld[k][2]);

            atomic_add(pgrad.dufldi[0], &dufld[i][0]);
            atomic_add(pgrad.dufldi[1], &dufld[i][1]);
            atomic_add(pgrad.dufldi[2], &dufld[i][2]);
            atomic_add(pgrad.dufldi[3], &dufld[i][3]);
            atomic_add(pgrad.dufldi[4], &dufld[i][4]);
            atomic_add(pgrad.dufldi[5], &dufld[i][5]);
            atomic_add(pgrad.dufldk[0], &dufld[k][0]);
            atomic_add(pgrad.dufldk[1], &dufld[k][1]);
            atomic_add(pgrad.dufldk[2], &dufld[k][2]);
            atomic_add(pgrad.dufldk[3], &dufld[k][3]);
            atomic_add(pgrad.dufldk[4], &dufld[k][4]);
            atomic_add(pgrad.dufldk[5], &dufld[k][5]);

            if CONSTEXPR (do_v) {
               real vxx = -xr * pgrad.frcx;
               real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               real vyy = -yr * pgrad.frcy;
               real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               real vzz = -zr * pgrad.frcz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
            }
            if CONSTEXPR (CFLX) {
               atomic_add(pota, pot, i);
               atomic_add(potb, pot, k);
            } // end if CFLX
         }    // end if (do_g)
      }       // end if (r2 <= off2)
   }          // end for (int ii)

   // torque

   if CONSTEXPR (do_g) {
      #pragma acc parallel loop independent async\
                  deviceptr(rpole,trqx,trqy,trqz,ufld,dufld)
      for (int i = 0; i < n; ++i) {
         real dix = rpole[i][MPL_PME_X];
         real diy = rpole[i][MPL_PME_Y];
         real diz = rpole[i][MPL_PME_Z];
         real qixx = rpole[i][MPL_PME_XX];
         real qixy = rpole[i][MPL_PME_XY];
         real qixz = rpole[i][MPL_PME_XZ];
         real qiyy = rpole[i][MPL_PME_YY];
         real qiyz = rpole[i][MPL_PME_YZ];
         real qizz = rpole[i][MPL_PME_ZZ];

         real tep1 = diz * ufld[i][1] - diy * ufld[i][2] + qixz * dufld[i][1] - qixy * dufld[i][3]
            + 2 * qiyz * (dufld[i][2] - dufld[i][5]) + (qizz - qiyy) * dufld[i][4];
         real tep2 = dix * ufld[i][2] - diz * ufld[i][0] - qiyz * dufld[i][1] + qixy * dufld[i][4]
            + 2 * qixz * (dufld[i][5] - dufld[i][0]) + (qixx - qizz) * dufld[i][3];
         real tep3 = diy * ufld[i][0] - dix * ufld[i][1] + qiyz * dufld[i][3] - qixz * dufld[i][4]
            + 2 * qixy * (dufld[i][0] - dufld[i][2]) + (qiyy - qixx) * dufld[i][1];

         trqx[i] += tep1;
         trqy[i] += tep2;
         trqz[i] += tep3;
      }
   }
}

void epolarAplusNonEwald_acc(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         // epolarAplus_acc1<calc::V0, NON_EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         epolarAplus_acc1<calc::V1, NON_EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         // epolarAplus_acc1<calc::V3, NON_EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         epolarAplus_acc1<calc::V4, NON_EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_acc1<calc::V5, NON_EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_acc1<calc::V6, NON_EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarAplus_acc1<calc::V0, NON_EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarAplus_acc1<calc::V1, NON_EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_acc1<calc::V3, NON_EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarAplus_acc1<calc::V4, NON_EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_acc1<calc::V5, NON_EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_acc1<calc::V6, NON_EWALD, 0>(uind);
      }
   }
}

void epolarAplusEwaldReal_acc(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         // epolarAplus_acc1<calc::V0, EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         epolarAplus_acc1<calc::V1, EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         // epolarAplus_acc1<calc::V3, EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         epolarAplus_acc1<calc::V4, EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_acc1<calc::V5, EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_acc1<calc::V6, EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarAplus_acc1<calc::V0, EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarAplus_acc1<calc::V1, EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_acc1<calc::V3, EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarAplus_acc1<calc::V4, EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_acc1<calc::V5, EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_acc1<calc::V6, EWALD, 0>(uind);
      }
   }
}
}
