#include "ff/modamoeba.h"
#include "seq/launch.h"
#include "seq/rotpole.h"

namespace tinker {
__global__
void chkpole_cu1(int n, real (*restrict pole)[MPL_TOTAL], LocalFrame* zaxis, const real* restrict x,
   const real* restrict y, const real* restrict z)
{
   for (int i = ITHREAD; i < n; i += STRIDE)
      chkpoleAtomI(i, pole, zaxis, x, y, z);
}

void chkpole_cu()
{
   launch_k1s(g::s0, n, chkpole_cu1, n, pole, zaxis, x, y, z);
}

void chkrepole_cu()
{
   launch_k1s(g::s0, n, chkpole_cu1, n, repole, zaxis, x, y, z);
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
   launch_k1s(g::s0, n, rotpole_cu1, n, rpole, pole, zaxis, x, y, z);
}

void rotrepole_cu()
{
   launch_k1s(g::s0, n, rotpole_cu1, n, rrepole, repole, zaxis, x, y, z);
}
}
