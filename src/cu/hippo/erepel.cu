#include "ff/dlmda.h"
#include "ff/hippo/erepel.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_repel.h"
#include "seq/triangle.h"

namespace tinker {
#include "erepel_cu1.cc"

template <class Ver>
static void erepel_cu2()
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);
   const SoftCoreLmda rs = repelSoft(Ver::e_dlmda1);

   int ngrid = gpuGridSize(BLOCK_DIM);
   erepel_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nrep, er, devdl_buf, d2evdl2_buf, vir_er,
      dvirdl_buf, derx, dery, derz, dfdlx, dfdly, dfdlz, cut, off, st.si2.bit0, nrepexclude, repexclude,
      repexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, trqx, trqy, trqz,
      dltrqx, dltrqy, dltrqz, rrepole, sizpr, elepr, dmppr, mut, vcouple, rs);
}

void erepel_cu(int vers)
{
   if (vers == calc::v0)
      erepel_cu2<calc::V0>();
   else if (vers == calc::v1)
      erepel_cu2<calc::V1>();
   else if (vers == calc::v3)
      erepel_cu2<calc::V3>();
   else if (vers == calc::v4)
      erepel_cu2<calc::V4>();
   else if (vers == calc::v5)
      erepel_cu2<calc::V5>();
   else if (vers == calc::v6)
      erepel_cu2<calc::V6>();
   else if (vers == calc::v7)
      erepel_cu2<calc::V7>();
   else if (vers == calc::v8)
      erepel_cu2<calc::V8>();
   else if (vers == calc::v9)
      erepel_cu2<calc::V9>();
   else if (vers == calc::v10)
      erepel_cu2<calc::V10>();
}
}
