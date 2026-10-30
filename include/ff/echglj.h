#pragma once
#include "ff/echarge.h"
#include "ff/evdw.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup chglj
/// \brief Whether the fused charge and Lennard-Jones kernel can replace
/// echarge() and evdw(); the analysis versions never use it.
bool echgljAvailable();
/// \ingroup chglj
void echgljData(RcOp);
/// \ingroup chglj
void echglj(int vers);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

// chglj
namespace tinker {
TINKER_EXTERN int ncvexclude;
TINKER_EXTERN int (*cvexclude)[2];
TINKER_EXTERN real (*cvexclude_scale)[2];

TINKER_EXTERN bool vdwpr_in_use;

TINKER_EXTERN int* mut_coalesced;     // n
TINKER_EXTERN real* chg_coalesced;    // n
TINKER_EXTERN real* radeps_coalesced; // 2*n
}
