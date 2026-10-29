#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modhippo.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_chgtrn.h"
#include "seq/pairchgtrnaplus.h"
#include "seq/triangle.h"
#include <cassert>

namespace tinker {
#include "echgtrn_cu1.cc"

template <class Ver, Chgtrn CT>
static void echgtrn_cu2()
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::CHGTRN);
   real off = switchOff(Switch::CHGTRN);
   real f = electric / dielec;

   // The lambda derivatives of charge transfer go with polarization.
   int ngrid = gpuGridSize(BLOCK_DIM);
   echgtrn_cu1<Ver, CT><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nct, ect, depdl_buf, d2epdl2_buf,
      vir_ect, dvirdl_buf, dectx, decty, dectz, dfdlx, dfdly, dfdlz, cut, off, st.si1.bit0, nmdwexclude, mdwexclude,
      mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, chgct, dmpct, f,
      emGroup(), ctScale());
}

template <Chgtrn CT>
static void echgtrn_cu3(int vers)
{
   if (vers == calc::v0)
      echgtrn_cu2<calc::V0, CT>();
   else if (vers == calc::v1)
      echgtrn_cu2<calc::V1, CT>();
   else if (vers == calc::v3)
      echgtrn_cu2<calc::V3, CT>();
   else if (vers == calc::v4)
      echgtrn_cu2<calc::V4, CT>();
   else if (vers == calc::v5)
      echgtrn_cu2<calc::V5, CT>();
   else if (vers == calc::v6)
      echgtrn_cu2<calc::V6, CT>();
   else if (vers == calc::v7)
      echgtrn_cu2<calc::V7, CT>();
   else if (vers == calc::v8)
      echgtrn_cu2<calc::V8, CT>();
   else if (vers == calc::v9)
      echgtrn_cu2<calc::V9, CT>();
   else if (vers == calc::v10)
      echgtrn_cu2<calc::V10, CT>();
}

void echgtrn_cu(int vers)
{
   assert(ctrntyp == Chgtrn::SEPARATE);
   echgtrn_cu3<Chgtrn::SEPARATE>(vers);
}

void echgtrnAplus_cu(int vers)
{
   assert(ctrntyp == Chgtrn::COMBINED);
   echgtrn_cu3<Chgtrn::COMBINED>(vers);
}
}
