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

   int ngrid = gpuGridSize(BLOCK_DIM);
   echgtrn_cu1<Ver, CT><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nct, ect, vir_ect, dectx, decty, dectz,
      cut, off, st.si1.bit0, nmdwexclude, mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, chgct, dmpct, f, mut, elam);
}

void echgtrn_cu(int vers)
{
   assert(ctrntyp == Chgtrn::SEPARATE);
   constexpr auto CT = Chgtrn::SEPARATE;
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
}

void echgtrnAplus_cu(int vers)
{
   assert(ctrntyp == Chgtrn::COMBINED);
   constexpr auto CT = Chgtrn::COMBINED;
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
}
}
