#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup charge
void echargeData(RcOp);

/// \ingroup charge
void echarge(int vers);

/// \ingroup charge
void echargeEwaldRecipSelf(int);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

// charge chgpot
namespace tinker {
/// \ingroup charge
/// \brief Magnitude of the partial charges (e-) of each **atom**.
/// \note Unlike Tinker, where only non-zero charges will be stored, this
/// array also includes zero charges.
TINKER_EXTERN real* pchg;
/// \ingroup charge
/// \brief The derivative of each partial charge with respect to elambda, the
/// unscaled charge times the slope of its group scale (dlambda.f:chgdlmda);
/// allocated with the electrostatic lambda derivatives only.
TINKER_EXTERN real* pchg_dq;
/// \ingroup charge
/// \brief The unscaled partial charges, kept while a ligand group can follow
/// lambda, from which pchg is rescaled; otherwise null.
TINKER_EXTERN real* pchgorig;

TINKER_EXTERN CountBuffer nec;
TINKER_EXTERN EnergyBuffer ec;
TINKER_EXTERN VirialBuffer vir_ec;
TINKER_EXTERN grad_prec* decx;
TINKER_EXTERN grad_prec* decy;
TINKER_EXTERN grad_prec* decz;
TINKER_EXTERN energy_prec energy_ec;
TINKER_EXTERN virial_prec virial_ec[9];

TINKER_EXTERN real ebuffer;
TINKER_EXTERN real c2scale, c3scale, c4scale, c5scale;
TINKER_EXTERN int ncexclude;
TINKER_EXTERN int (*cexclude)[2];
TINKER_EXTERN real* cexclude_scale;
}
