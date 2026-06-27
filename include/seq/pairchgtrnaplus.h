#pragma once
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <bool DO_G>
SEQ_CUDA
void pair_chgtrn_aplus(real r,
                       real cut,
                       real off,
                       real mscale,
                       real f,
                       real alphai,
                       real chgi,
                       real alphak,
                       real chgk,
                       real elambda,
                       e_prec& restrict e,
                       e_prec& restrict de)
{
   f *= mscale;
   real alphaik = 0.5 * (alphai + alphak);
   real chgik = REAL_SQRT(chgi * chgk);
   e = -chgik * REAL_EXP(-alphaik * r);
   e *= f * elambda;
   if CONSTEXPR (DO_G) {
      de = alphaik * chgik * REAL_EXP(-alphaik * r);
      de *= f * elambda;
   }
   if (r > cut) {
      real taper, dtaper;
      switchTaper5<DO_G>(r, cut, off, taper, dtaper);
      if CONSTEXPR (DO_G)
         de = e * dtaper + de * taper;
      e *= taper;
   }
}
}
