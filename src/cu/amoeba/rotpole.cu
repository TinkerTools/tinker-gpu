#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "seq/launch.h"
#include "seq/rotpole.h"

namespace tinker {
__global__
void chkpole_cu1(int n, real (*restrict pole)[MPL_TOTAL], real (*restrict pole2)[MPL_TOTAL],
   LocalFrame* zaxis, const real* restrict x, const real* restrict y, const real* restrict z)
{
   for (int i = ITHREAD; i < n; i += STRIDE)
      chkpoleAtomI(i, pole, pole2, zaxis, x, y, z);
}

void chkpole_cu()
{
   launch_k1s(g::s0, n, chkpole_cu1, n, pole, poleorig, zaxis, x, y, z);
}

void chkrepole_cu()
{
   launch_k1s(g::s0, n, chkpole_cu1, n, repole, nullptr, zaxis, x, y, z);
}
}

namespace tinker {
__global__
void rotpole_cu1(int n, real (*restrict rpole)[MPL_TOTAL], const real (*restrict pole)[MPL_TOTAL],
   const LocalFrame* restrict zaxis, const real* restrict x, const real* restrict y, const real* restrict z)
{
   for (int i = ITHREAD; i < n; i += STRIDE)
      rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
}

void rotpole_cu()
{
   if (use_emast) {
      launch_k1s(g::s0, n, rotpole_cu1, n, rpole, poleorig, zaxis, x, y, z);
   } else {
      launch_k1s(g::s0, n, rotpole_cu1, n, rpole, pole, zaxis, x, y, z);
   }
}

__global__
static void rotpoleState_cu1(int n, real (*restrict rpole)[MPL_TOTAL],
   const real (*restrict pole)[MPL_TOTAL], const LocalFrame* restrict zaxis, const real* restrict x,
   const real* restrict y, const real* restrict z, RdtMask mask, const int* restrict group)
{
   unsigned active_mask = static_cast<unsigned>(mask);
   for (int i = ITHREAD; i < n; i += STRIDE) {
      unsigned atom_mask = static_cast<unsigned>(RdtMask::ENV);
      if (group[i] == 1)
         atom_mask = static_cast<unsigned>(RdtMask::LIGA);
      else if (group[i] == 2)
         atom_mask = static_cast<unsigned>(RdtMask::LIGB);
      if (active_mask & atom_mask)
         rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
      else
         for (int j = 0; j < MPL_TOTAL; ++j)
            rpole[i][j] = 0;
   }
}

void rotpoleState_cu(RdtMask mask, const int* group)
{
   launch_k1s(g::s0, n, rotpoleState_cu1, n, rpole, poleorig, zaxis, x, y, z, mask, group);
}

void rotrepole_cu()
{
   launch_k1s(g::s0, n, rotpole_cu1, n, rrepole, repole, zaxis, x, y, z);
}

__global__
static void mpoleScale_cu1(int n, real (*restrict pole)[MPL_TOTAL],
   const real (*restrict poleorig)[MPL_TOTAL], const int* restrict mut, real factor)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (mut[i]) {
         for (int j = 0; j < MPL_TOTAL; ++j)
            pole[i][j] = factor * poleorig[i][j];
      }
   }
}

void mpoleScale_cu(real factor)
{
   launch_k1s(g::s0, n, mpoleScale_cu1, n, pole, poleorig, mut, factor);
}
}
