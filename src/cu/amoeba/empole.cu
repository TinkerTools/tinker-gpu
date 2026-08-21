#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/emselfamoeba.h"
#include "seq/launch.h"
#include "seq/pair_mpole.h"
#include "seq/triangle.h"
#include <tinker/detail/extfld.hh>

namespace tinker {
#include "empole_cu1.cc"
#include "empoledlmda_cu1.cc"

template <class Ver, class ETYP, class LTYP>
static void empole_cu()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;

   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;

      if CONSTEXPR (do_e) {
         if CONSTEXPR (eq<LTYP, DLMDA>()) {
            launch_k1b(g::s0, n, empoleSelfDlmda_cu<Ver>, //
               nem, em, demdl_buf, d2emdl2_buf, rpole, mut, n, f, aewald, elam, deldlmda, d2eldlmda2);
         } else {
            launch_k1b(g::s0, n, empoleSelf_cu<do_a>, //
               nem, em, rpole, n, f, aewald);
         }
      }
   }
   int ngrid = gpuGridSize(BLOCK_DIM);
   if CONSTEXPR (eq<LTYP, DLMDA>()) {
      empoledlmda_cu1<Ver, ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, demdl_buf, d2emdl2_buf,
         vir_em, dvirdl_buf, demx, demy, demz, dfdlx, dfdly, dfdlz, off, st.si1.bit0, nmdpuexclude,
         mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, mut, f, aewald, elam, deldlmda, d2eldlmda2);
   } else {
      empole_cu1<Ver, ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, vir_em, demx, demy, demz,
         off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
         st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, f, aewald);
   }
}

void empoleNonEwald_cu(int vers)
{
   if (use_emast) {
      if (vers == calc::v0) {
         empole_cu<calc::V0, NON_EWALD,DLMDA>();
      } else if (vers == calc::v1) {
         empole_cu<calc::V1, NON_EWALD,DLMDA>();
      } else if (vers == calc::v3) {
         empole_cu<calc::V3, NON_EWALD,DLMDA>();
      } else if (vers == calc::v4) {
         empole_cu<calc::V4, NON_EWALD,DLMDA>();
      } else if (vers == calc::v5) {
         empole_cu<calc::V5, NON_EWALD,DLMDA>();
      } else if (vers == calc::v6) {
         empole_cu<calc::V6, NON_EWALD,DLMDA>();
      } else if (vers == calc::v7) {
         empole_cu<calc::V7, NON_EWALD,DLMDA>();
      } else if (vers == calc::v8) {
         empole_cu<calc::V8, NON_EWALD,DLMDA>();
      } else if (vers == calc::v9) {
         empole_cu<calc::V9, NON_EWALD,DLMDA>();
      } else if (vers == calc::v10) {
         empole_cu<calc::V10, NON_EWALD,DLMDA>();
      }
   } else {
      if (vers == calc::v0) {
         empole_cu<calc::V0, NON_EWALD,NON_DLMDA>();
      } else if (vers == calc::v1) {
         empole_cu<calc::V1, NON_EWALD,NON_DLMDA>();
      } else if (vers == calc::v3) {
         empole_cu<calc::V3, NON_EWALD,NON_DLMDA>();
      } else if (vers == calc::v4) {
         empole_cu<calc::V4, NON_EWALD,NON_DLMDA>();
      } else if (vers == calc::v5) {
         empole_cu<calc::V5, NON_EWALD,NON_DLMDA>();
      } else if (vers == calc::v6) {
         empole_cu<calc::V6, NON_EWALD,NON_DLMDA>();
      }
   }
}

void empoleEwaldRealSelf_cu(int vers)
{
   if (use_emast) {
      if (vers == calc::v0) {
         empole_cu<calc::V0, EWALD,DLMDA>();
      } else if (vers == calc::v1) {
         empole_cu<calc::V1, EWALD,DLMDA>();
      } else if (vers == calc::v3) {
         empole_cu<calc::V3, EWALD,DLMDA>();
      } else if (vers == calc::v4) {
         empole_cu<calc::V4, EWALD,DLMDA>();
      } else if (vers == calc::v5) {
         empole_cu<calc::V5, EWALD,DLMDA>();
      } else if (vers == calc::v6) {
         empole_cu<calc::V6, EWALD,DLMDA>();
      } else if (vers == calc::v7) {
         empole_cu<calc::V7, EWALD,DLMDA>();
      } else if (vers == calc::v8) {
         empole_cu<calc::V8, EWALD,DLMDA>();
      } else if (vers == calc::v9) {
         empole_cu<calc::V9, EWALD,DLMDA>();
      } else if (vers == calc::v10) {
         empole_cu<calc::V10, EWALD,DLMDA>();
      }
   } else {
      if (vers == calc::v0) {
         empole_cu<calc::V0, EWALD,NON_DLMDA>();
      } else if (vers == calc::v1) {
         empole_cu<calc::V1, EWALD,NON_DLMDA>();
      } else if (vers == calc::v3) {
         empole_cu<calc::V3, EWALD,NON_DLMDA>();
      } else if (vers == calc::v4) {
         empole_cu<calc::V4, EWALD,NON_DLMDA>();
      } else if (vers == calc::v5) {
         empole_cu<calc::V5, EWALD,NON_DLMDA>();
      } else if (vers == calc::v6) {
         empole_cu<calc::V6, EWALD,NON_DLMDA>();
      }
   }
}

template <class Ver>
__global__
static void exfieldDipole_cu1(CountBuffer restrict nem, EnergyBuffer restrict em, VirialBuffer vir_em,
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz, real* restrict trqx,
   real* restrict trqy, real* restrict trqz, int n, real f, real ef1, real ef2, real ef3,
   const real (*restrict rpole)[10], const real* restrict x, const real* restrict y, const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii];
      real ci = rpole[ii][0], dix = rpole[ii][1], diy = rpole[ii][2], diz = rpole[ii][3];

      if CONSTEXPR (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * (ci * phi + dix * ef1 + diy * ef2 + diz * ef3);
         atomic_add(e, em, ithread);
         if CONSTEXPR (do_a) {
            if (e != 0)
               atomic_add(1, nem, ithread);
         }
      }
      if CONSTEXPR (do_g) {
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
         if CONSTEXPR (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, ithread);
         }
      }
   }
}

template <class Ver>
static void exfieldDipole_cu2()
{
   real f = electric / dielec;
   real ef1 = extfld::texfld[0], ef2 = extfld::texfld[1], ef3 = extfld::texfld[2];
   launch_k1b(g::s0, n, exfieldDipole_cu1<Ver>, nem, em, vir_em, demx, demy, demz, trqx, trqy, trqz, n, f, ef1, ef2,
      ef3, rpole, x, y, z);
}

void exfieldDipole_cu(int vers)
{
   if (vers == calc::v0)
      exfieldDipole_cu2<calc::V0>();
   else if (vers == calc::v1)
      exfieldDipole_cu2<calc::V1>();
   else if (vers == calc::v3)
      exfieldDipole_cu2<calc::V3>();
   else if (vers == calc::v4)
      exfieldDipole_cu2<calc::V4>();
   else if (vers == calc::v5)
      exfieldDipole_cu2<calc::V5>();
   else if (vers == calc::v6)
      exfieldDipole_cu2<calc::V6>();
}

template <class Ver>
__global__
static void exfieldDipoleDlmda_cu1(CountBuffer restrict nem, EnergyBuffer restrict em, EnergyBuffer restrict demdl,
   VirialBuffer vir_em, VirialBuffer restrict demvirdl,                                //
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz,       //
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz, //
   real* restrict trqx, real* restrict trqy, real* restrict trqz,                      //
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz,                //
   int n, real f, real ef1, real ef2, real ef3, const real (*restrict rpole)[10],
   const int* restrict mut, real elambda, const real* restrict x, const real* restrict y, const real* restrict z,
   EnergyBuffer restrict d2emdl2, real deldl, real d2eldl2)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii];
      real ci = rpole[ii][0], dix = rpole[ii][1], diy = rpole[ii][2], diz = rpole[ii][3];

      // the mutated sites carry multipoles scaled by elambda
      bool muti = mut[ii];
      real s = muti ? elambda : 1;

      if CONSTEXPR (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * (ci * phi + dix * ef1 + diy * ef2 + diz * ef3);
         if (muti) {
            if CONSTEXPR (do_dl1)
               atomic_add(deldl * e, demdl, ithread);
            if CONSTEXPR (do_dl2)
               atomic_add(d2eldl2 * e, d2emdl2, ithread);
         }
         e *= s;
         atomic_add(e, em, ithread);
         if CONSTEXPR (do_a) {
            if (e != 0)
               atomic_add(1, nem, ithread);
         }
      }
      if CONSTEXPR (do_g) {
         // unscaled torque due to the dipole
         real tx = f * (diy * ef3 - diz * ef2);
         real ty = f * (diz * ef1 - dix * ef3);
         real tz = f * (dix * ef2 - diy * ef1);
         atomic_add(s * tx, trqx, ii);
         atomic_add(s * ty, trqy, ii);
         atomic_add(s * tz, trqz, ii);
         // unscaled gradient and virial due to the monopole
         real frx = -f * ef1 * ci;
         real fry = -f * ef2 * ci;
         real frz = -f * ef3 * ci;
         atomic_add(s * frx, demx, ii);
         atomic_add(s * fry, demy, ii);
         atomic_add(s * frz, demz, ii);
         if (muti) {
            if CONSTEXPR (do_tdl) {
               atomic_add(deldl * tx, dltrqx, ii);
               atomic_add(deldl * ty, dltrqy, ii);
               atomic_add(deldl * tz, dltrqz, ii);
            }
            if CONSTEXPR (do_gdl) {
               atomic_add(deldl * frx, dfmdlx, ii);
               atomic_add(deldl * fry, dfmdly, ii);
               atomic_add(deldl * frz, dfmdlz, ii);
            }
         }
         if CONSTEXPR (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(s * vxx, s * vxy, s * vxz, s * vyy, s * vyz, s * vzz, vir_em, ithread);
            if CONSTEXPR (do_vdl) {
               if (muti) {
                  atomic_add(deldl * vxx, deldl * vxy, deldl * vxz, deldl * vyy, deldl * vyz, deldl * vzz,
                     demvirdl, ithread);
               }
            }
         }
      }
   }
}

template <class Ver>
static void exfieldDipoleDlmda_cu2()
{
   real f = electric / dielec;
   real ef1 = extfld::texfld[0], ef2 = extfld::texfld[1], ef3 = extfld::texfld[2];
   launch_k1b(g::s0, n, exfieldDipoleDlmda_cu1<Ver>, nem, em, demdl_buf, vir_em, dvirdl_buf, demx, demy, demz,
      dfdlx, dfdly, dfdlz, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, n, f, ef1, ef2, ef3, rpole, mut, elam,
      x, y, z, d2emdl2_buf, deldlmda, d2eldlmda2);
}

void exfieldDipoleDlmda_cu(int vers)
{
   if (vers == calc::v0)
      exfieldDipoleDlmda_cu2<calc::V0>();
   else if (vers == calc::v1)
      exfieldDipoleDlmda_cu2<calc::V1>();
   else if (vers == calc::v3)
      exfieldDipoleDlmda_cu2<calc::V3>();
   else if (vers == calc::v4)
      exfieldDipoleDlmda_cu2<calc::V4>();
   else if (vers == calc::v5)
      exfieldDipoleDlmda_cu2<calc::V5>();
   else if (vers == calc::v6)
      exfieldDipoleDlmda_cu2<calc::V6>();
   else if (vers == calc::v7)
      exfieldDipoleDlmda_cu2<calc::V7>();
   else if (vers == calc::v8)
      exfieldDipoleDlmda_cu2<calc::V8>();
   else if (vers == calc::v9)
      exfieldDipoleDlmda_cu2<calc::V9>();
   else if (vers == calc::v10)
      exfieldDipoleDlmda_cu2<calc::V10>();
}

__global__
static void extfieldModifyDField_cu1(real (*restrict field)[3], real (*restrict fieldp)[3], int n, real ex1, real ex2,
   real ex3)
{
   int ithread = ITHREAD;
   if (fieldp) {
      for (int i = ithread; i < n; i += STRIDE) {
         field[i][0] += ex1;
         field[i][1] += ex2;
         field[i][2] += ex3;
         fieldp[i][0] += ex1;
         fieldp[i][1] += ex2;
         fieldp[i][2] += ex3;
      }
   } else {
      for (int i = ithread; i < n; i += STRIDE) {
         field[i][0] += ex1;
         field[i][1] += ex2;
         field[i][2] += ex3;
      }
   }
}

void extfieldModifyDField_cu(real (*field)[3], real (*fieldp)[3])
{
   real ex1 = extfld::texfld[0];
   real ex2 = extfld::texfld[1];
   real ex3 = extfld::texfld[2];
   launch_k1b(g::s0, n, extfieldModifyDField_cu1, field, fieldp, n, ex1, ex2, ex3);
}
}
