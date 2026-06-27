#include "ff/cumodamoeba.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/damp.h"
#include "seq/launch.h"
#include "seq/triangle.h"

namespace tinker {
__global__
void sparsePrecond_cu0(const real (*restrict rsd)[3], const real (*restrict rsdp)[3], real (*restrict zrsd)[3],
   real (*restrict zrsdp)[3], const real* restrict polarity, int n, real udiag)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real poli = udiag * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         zrsd[i][j] = poli * rsd[i][j];
         zrsdp[i][j] = poli * rsdp[i][j];
      }
   }
}

#include "sparsePrecond_cu1.cc"

void sparsePrecondApply_cu(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3])
{
   const auto& st = *uspatial_v2_unit;
   real off = switchOff(Switch::USOLVE);
   off = off + st.buffer;

   launch_k1s(g::s0, n, sparsePrecond_cu0, //
      rsd, rsdp, zrsd, zrsdp, polarity, n, udiag);
   int ngrid = gpuGridSize(BLOCK_DIM);
   ngrid *= BLOCK_DIM;
   int nparallel = std::max(st.niak, st.nakpl) * WARP_SIZE;
   nparallel = std::max(nparallel, ngrid);
   launch_k1s(g::s0, nparallel, sparsePrecond_cu1, //
      st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niak, st.iak, st.lst, rsd, rsdp, zrsd, zrsdp, polarity);
}
}
