#pragma once
#include "ff/dlmda.h"
#include "seq/seq.h"

namespace tinker {
SEQ_CUDA
inline bool rdtActive(RdtMask mask, int group)
{
   auto bits = static_cast<unsigned>(mask);
   return bits & (1u << group);
}

SEQ_CUDA
inline bool rdtPairActive(RdtMask mask, int groupi, int groupk)
{
   return rdtActive(mask, groupi) && rdtActive(mask, groupk);
}
}
