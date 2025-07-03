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

template <class Ver, class ETYP>
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
         launch_k1b(g::s0, n, empoleSelf_cu<do_a>, //
            nem, em, rpole, n, f, aewald);
      }
   }
   int ngrid = gpuGridSize(BLOCK_DIM);
   empole_cu1<Ver, ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, vir_em, demx, demy, demz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, f, aewald);
}

void empoleNonEwald_cu(int vers)
{
   if (vers == calc::v0) {
      empole_cu<calc::V0, NON_EWALD>();
   } else if (vers == calc::v1) {
      empole_cu<calc::V1, NON_EWALD>();
   } else if (vers == calc::v3) {
      empole_cu<calc::V3, NON_EWALD>();
   } else if (vers == calc::v4) {
      empole_cu<calc::V4, NON_EWALD>();
   } else if (vers == calc::v5) {
      empole_cu<calc::V5, NON_EWALD>();
   } else if (vers == calc::v6) {
      empole_cu<calc::V6, NON_EWALD>();
   }
}

void empoleEwaldRealSelf_cu(int vers)
{
   if (vers == calc::v0) {
      empole_cu<calc::V0, EWALD>();
   } else if (vers == calc::v1) {
      empole_cu<calc::V1, EWALD>();
   } else if (vers == calc::v3) {
      empole_cu<calc::V3, EWALD>();
   } else if (vers == calc::v4) {
      empole_cu<calc::V4, EWALD>();
   } else if (vers == calc::v5) {
      empole_cu<calc::V5, EWALD>();
   } else if (vers == calc::v6) {
      empole_cu<calc::V6, EWALD>();
   }
}

__global__
static void exfieldDipole_cu1(CountBuffer restrict nem, EnergyBuffer restrict em, VirialBuffer vir_em,
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz, real* restrict trqx,
   real* restrict trqy, real* restrict trqz, int vers, int n, real f, real ef1, real ef2, real ef3,
   const real (*restrict rpole)[10], const real* restrict x, const real* restrict y, const real* restrict z)
{
   bool do_e = vers & calc::energy;
   bool do_a = vers & calc::analyz;
   bool do_g = vers & calc::grad;
   bool do_v = vers & calc::virial;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii];
      real ci = rpole[ii][0], dix = rpole[ii][1], diy = rpole[ii][2], diz = rpole[ii][3];

      if (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * (ci * phi + dix * ef1 + diy * ef2 + diz * ef3);
         atomic_add(e, em, ithread);
         if (do_a)
            atomic_add(1, nem, ithread);
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
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, ithread);
         }
      }
   }
}

void exfieldDipole_cu(int vers)
{
   real f = electric / dielec;
   real ef1 = extfld::exfld[0], ef2 = extfld::exfld[1], ef3 = extfld::exfld[2];
   launch_k1s(g::s0, n, exfieldDipole_cu1, nem, em, vir_em, demx, demy, demz, trqx, trqy, trqz, vers, n, f, ef1, ef2,
      ef3, rpole, x, y, z);
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
   real ex1 = extfld::exfld[0];
   real ex2 = extfld::exfld[1];
   real ex3 = extfld::exfld[2];
   launch_k1b(g::s0, n, extfieldModifyDField_cu1, field, fieldp, n, ex1, ex2, ex3);
}
}
