#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/precision.h"

namespace tinker {
__global__
static void epolarTorque_cu(real* restrict trqx,
                            real* restrict trqy,
                            real* restrict trqz,
                            int n,
                            const real (*restrict rpole)[10],
                            const real (*restrict ufld)[3],
                            const real (*restrict dufld)[6])
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];

      real tep1 = diz * ufld[i][1] - diy * ufld[i][2] + qixz * dufld[i][1] - qixy * dufld[i][3] + 2 * qiyz * (dufld[i][2] - dufld[i][5])
         + (qizz - qiyy) * dufld[i][4];
      real tep2 = dix * ufld[i][2] - diz * ufld[i][0] - qiyz * dufld[i][1] + qixy * dufld[i][4] + 2 * qixz * (dufld[i][5] - dufld[i][0])
         + (qixx - qizz) * dufld[i][3];
      real tep3 = diy * ufld[i][0] - dix * ufld[i][1] + qiyz * dufld[i][3] - qixz * dufld[i][4] + 2 * qixy * (dufld[i][0] - dufld[i][2])
         + (qiyy - qixx) * dufld[i][1];

      trqx[i] += tep1;
      trqy[i] += tep2;
      trqz[i] += tep3;
   }
}
}
