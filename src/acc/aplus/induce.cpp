#include "ff/atom.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "seq/add.h"
#include "seq/damp.h"
#include "tool/gpucard.h"

namespace tinker {
#define APPLY_DPTRS rsd, zrsd, x, y, z, polarity, pdamp, thole
void sparsePrecondApply3_acc(const real (*rsd)[3], real (*zrsd)[3])
{
   #pragma acc parallel loop independent async\
               deviceptr(polarity,rsd,zrsd)
   for (int i = 0; i < n; ++i) {
      real poli = udiag * polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j)
         zrsd[i][j] = poli * rsd[i][j];
   }

   const int maxnlst = ulist_unit->maxnlst;
   const auto* ulst = ulist_unit.deviceptr();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(APPLY_DPTRS,ulst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real pdi = pdamp[i];
      real pti = thole[i];
      real poli = polarity[i];

      int nulsti = ulst->nlst[i];
      int base = i * maxnlst;
      real gxi = 0, gyi = 0, gzi = 0;

      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nulsti; ++kk) {
         int k = ulst->lst[base + kk];

         real pdk = pdamp[k];
         real ptk = thole[k];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         real r = REAL_SQRT(r2);

         real scale3, scale5;
         damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

         real polik = poli * polarity[k];
         real rr3 = scale3 * polik * REAL_RECIP(r * r2);
         real rr5 = 3 * scale5 * polik * REAL_RECIP(r * r2 * r2);

         real m0 = rr5 * xr * xr - rr3;
         real m1 = rr5 * xr * yr;
         real m2 = rr5 * xr * zr;
         real m3 = rr5 * yr * yr - rr3;
         real m4 = rr5 * yr * zr;
         real m5 = rr5 * zr * zr - rr3;

         gxi += m0 * rsd[k][0] + m1 * rsd[k][1] + m2 * rsd[k][2];
         gyi += m1 * rsd[k][0] + m3 * rsd[k][1] + m4 * rsd[k][2];
         gzi += m2 * rsd[k][0] + m4 * rsd[k][1] + m5 * rsd[k][2];

         atomic_add(m0 * rsd[i][0] + m1 * rsd[i][1] + m2 * rsd[i][2], &zrsd[k][0]);
         atomic_add(m1 * rsd[i][0] + m3 * rsd[i][1] + m4 * rsd[i][2], &zrsd[k][1]);
         atomic_add(m2 * rsd[i][0] + m4 * rsd[i][1] + m5 * rsd[i][2], &zrsd[k][2]);
      }

      atomic_add(gxi, &zrsd[i][0]);
      atomic_add(gyi, &zrsd[i][1]);
      atomic_add(gzi, &zrsd[i][2]);
   }

   #pragma acc parallel loop async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(APPLY_DPTRS,uexclude,uexclude_scale)
   for (int ii = 0; ii < nuexclude; ++ii) {
      int i = uexclude[ii][0];
      int k = uexclude[ii][1];
      real uscale = uexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real pdi = pdamp[i];
      real pti = thole[i];
      real poli = polarity[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      real r = REAL_SQRT(r2);

      real scale3, scale5;
      damp_thole2(r, pdi, pti, pdamp[k], thole[k], scale3, scale5);
      scale3 *= uscale;
      scale5 *= uscale;

      real polik = poli * polarity[k];
      real rr3 = scale3 * polik * REAL_RECIP(r * r2);
      real rr5 = 3 * scale5 * polik * REAL_RECIP(r * r2 * r2);

      real m0 = rr5 * xr * xr - rr3;
      real m1 = rr5 * xr * yr;
      real m2 = rr5 * xr * zr;
      real m3 = rr5 * yr * yr - rr3;
      real m4 = rr5 * yr * zr;
      real m5 = rr5 * zr * zr - rr3;

      atomic_add(m0 * rsd[k][0] + m1 * rsd[k][1] + m2 * rsd[k][2], &zrsd[i][0]);
      atomic_add(m1 * rsd[k][0] + m3 * rsd[k][1] + m4 * rsd[k][2], &zrsd[i][1]);
      atomic_add(m2 * rsd[k][0] + m4 * rsd[k][1] + m5 * rsd[k][2], &zrsd[i][2]);

      atomic_add(m0 * rsd[i][0] + m1 * rsd[i][1] + m2 * rsd[i][2], &zrsd[k][0]);
      atomic_add(m1 * rsd[i][0] + m3 * rsd[i][1] + m4 * rsd[i][2], &zrsd[k][1]);
      atomic_add(m2 * rsd[i][0] + m4 * rsd[i][1] + m5 * rsd[i][2], &zrsd[k][2]);
   }
}
}
