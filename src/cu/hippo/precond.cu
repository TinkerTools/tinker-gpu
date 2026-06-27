#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/damp.h"
#include "seq/damp_hippo.h"
#include "seq/launch.h"
#include "seq/triangle.h"

namespace tinker {
__global__
void sparsePrecond_cu3(const real (*restrict rsd)[3], real (*restrict zrsd)[3], const real* restrict polarity, int n,
   real udiag)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real poli = udiag * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         zrsd[i][j] = poli * rsd[i][j];
   }
}

#include "sparsePrecond_cu4.cc"

void sparsePrecondApply2_cu(const real (*rsd)[3], real (*zrsd)[3])
{
   const auto& st = *uspatial_v2_unit;
   const real off = switchOff(Switch::USOLVE) + st.buffer;

   launch_k1s(g::s0, n, sparsePrecond_cu3, //
      rsd, zrsd, polarity, n, udiag);

   int ngrid = gpuGridSize(BLOCK_DIM);
   sparsePrecond_cu4<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nwexclude, wexclude,
      wexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, rsd, zrsd, palpha,
      polarity);
}

#include "sparsePrecond_cu6.cc"

void sparsePrecondApply3_cu(const real (*rsd)[3], real (*zrsd)[3])
{
   const auto& st = *uspatial_v2_unit;
   const real off = switchOff(Switch::USOLVE) + st.buffer;

   launch_k1s(g::s0, n, sparsePrecond_cu3, //
      rsd, zrsd, polarity, n, udiag);

   int ngrid = gpuGridSize(BLOCK_DIM);
   sparsePrecond_cu6<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nuexclude, uexclude,
      uexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, rsd, zrsd, pdamp, thole,
      polarity);
}
}
