#pragma once
#include "ff/evdw.h"
#include "seq/seq.h"

namespace tinker {
/**
 * \ingroup vdw
 */
SEQ_ROUTINE
inline real pair_vlambda(real vlam, Vdw vcouple, int imut, int kmut)
{
   real vlambda = 1;
   if (vcouple == Vdw::DECOUPLE) {
      vlambda = (imut == kmut ? 1 : vlam);
   } else if (vcouple == Vdw::ANNIHILATE) {
      vlambda = (imut || kmut ? vlam : 1);
   }
   return vlambda;
}

/**
 * \ingroup vdw
 * \brief Whether a pair is scaled by the van der Waals lambda. A mutated pair
 * at lambda one still has a lambda derivative, so this is not vlambda < 1.
 */
SEQ_ROUTINE
inline bool pair_mutik(Vdw vcouple, int imut, int kmut)
{
   if (vcouple == Vdw::DECOUPLE)
      return imut != kmut;
   else if (vcouple == Vdw::ANNIHILATE)
      return imut || kmut;
   return false;
}
}
