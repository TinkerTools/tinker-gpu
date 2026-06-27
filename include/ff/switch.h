#pragma once
#include "ff/precision.h"

namespace tinker {
/// \ingroup ff
enum class Switch
{
   DEFAULT,
   VDW,
   REPULS,
   DISP,
   CHARGE,
   CHGDPL,
   DIPOLE,
   MPOLE,
   CHGTRN,
   EWALD,
   DEWALD,
   USOLVE,
   GKV,
   GKSA,
   NN,
};

/// \ingroup ff
/// \return Distance at which switching of the potential begins.
real switchCut(Switch mode);

/// \ingroup ff
/// \return Distance at which the potential energy goes to zero.
real switchOff(Switch mode);
}
