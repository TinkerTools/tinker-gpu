#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "seq/launch.h"
#include "seq/rotpole.h"

namespace tinker {
__global__
void chkpole_cu1(int n, real (*restrict pole)[MPL_TOTAL], real (*restrict poleorig)[MPL_TOTAL],
   real (*restrict repole)[MPL_TOTAL], LocalFrame* zaxis, const real* restrict x, const real* restrict y,
   const real* restrict z)
{
   for (int i = ITHREAD; i < n; i += STRIDE)
      chkpoleAtomI(i, pole, poleorig, repole, zaxis, x, y, z);
}

void chkpole_cu(real (*pole)[MPL_TOTAL], real (*poleorig)[MPL_TOTAL], real (*repole)[MPL_TOTAL])
{
   launch_k1s(g::s0, n, chkpole_cu1, n, pole, poleorig, repole, zaxis, x, y, z);
}
}

namespace tinker {
__global__
void rotpole_cu1(int n, real (*restrict rpole)[MPL_TOTAL], const real (*restrict pole)[MPL_TOTAL],
   const LocalFrame* restrict zaxis, const real* restrict x, const real* restrict y, const real* restrict z,
   const real* restrict dq)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
      if (dq)
         rpole[i][MPL_PME_0] += dq[i];
   }
}

void rotpole_cu(bool use_orig)
{
   if (use_orig) {
      // the unscaled monopoles carry the unscaled flux change (rotpole.f)
      const real* dq = use(Potent::CHGFLX) ? pdelta : nullptr;
      launch_k1s(g::s0, n, rotpole_cu1, n, rpole, poleorig, zaxis, x, y, z, dq);
   } else {
      launch_k1s(g::s0, n, rotpole_cu1, n, rpole, pole, zaxis, x, y, z, nullptr);
   }
}

__global__
static void rotpoleState_cu1(int n, real (*restrict rpole)[MPL_TOTAL],
   const real (*restrict pole)[MPL_TOTAL], const LocalFrame* restrict zaxis, const real* restrict x,
   const real* restrict y, const real* restrict z, RdtMask mask, const int* restrict group,
   const real* restrict dq, real* restrict pcorest, real* restrict pvalst, const real* restrict pcoreorig,
   const real* restrict pvalorig)
{
   unsigned active_mask = static_cast<unsigned>(mask);
   for (int i = ITHREAD; i < n; i += STRIDE) {
      unsigned atom_mask = static_cast<unsigned>(RdtMask::ENV);
      if (group[i] == 1)
         atom_mask = static_cast<unsigned>(RdtMask::LIGA);
      else if (group[i] == 2)
         atom_mask = static_cast<unsigned>(RdtMask::LIGB);
      // an active site carries its unscaled multipoles and flux change, and
      // the charge penetration charges that go with them
      if (active_mask & atom_mask) {
         rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
         real d = dq ? dq[i] : 0;
         rpole[i][MPL_PME_0] += d;
         if (pcorest) {
            pcorest[i] = pcoreorig[i];
            pvalst[i] = pvalorig[i] + d;
         }
      } else {
         for (int j = 0; j < MPL_TOTAL; ++j)
            rpole[i][j] = 0;
         if (pcorest) {
            pcorest[i] = 0;
            pvalst[i] = 0;
         }
      }
   }
}

void rotpoleState_cu(RdtMask mask, const int* group)
{
   const real* dq = use(Potent::CHGFLX) ? pdelta : nullptr;
   launch_k1s(g::s0, n, rotpoleState_cu1, n, rpole, poleorig, zaxis, x, y, z, mask, group, dq, pcorest, pvalst,
      pcoreorig, pvalorig);
}

void rotrepole_cu()
{
   launch_k1s(g::s0, n, rotpole_cu1, n, rrepole, repole, zaxis, x, y, z, nullptr);
}

__global__
static void mpoleScale_cu1(int n, real (*restrict pole)[MPL_TOTAL],
   const real (*restrict poleorig)[MPL_TOTAL], const int* restrict grp, GrpScale sc)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (grp[i]) {
         real factor = sc.s[grp[i]];
         for (int j = 0; j < MPL_TOTAL; ++j)
            pole[i][j] = factor * poleorig[i][j];
      }
   }
}

void mpoleScale_cu(const int* grp, GrpScale sc)
{
   launch_k1s(g::s0, n, mpoleScale_cu1, n, pole, poleorig, grp, sc);
}

// The charge penetration and charge flux parameters of each mutated site,
// scaled with its multipoles (mutate.f:altelec, altpolr): core and valence
// charges, the monopole the flux change is added to, and the flux scale.
__global__
static void mpoleScaleChgpen_cu1(int n, real* restrict pcore, real* restrict pval0, real* restrict pval,
   real* restrict mono0, real* restrict fluxsc, const real* restrict pcoreorig, const real* restrict pvalorig,
   const real (*restrict poleorig)[MPL_TOTAL], const int* restrict grp, GrpScale sc)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (grp[i]) {
         real s = sc.s[grp[i]];
         if (pcore) {
            pcore[i] = s * pcoreorig[i];
            pval0[i] = s * pvalorig[i];
            pval[i] = s * pvalorig[i];
         }
         if (mono0) {
            mono0[i] = s * poleorig[i][MPL_PME_0];
            fluxsc[i] = s;
         }
      }
   }
}

void mpoleScaleChgpen_cu(const int* grp, GrpScale sc, bool chgpen, bool cflux)
{
   launch_k1s(g::s0, n, mpoleScaleChgpen_cu1, n, chgpen ? pcore : nullptr, pval0, pval, cflux ? mono0 : nullptr,
      fluxsc, pcoreorig, pvalorig, poleorig, grp, sc);
}
}
