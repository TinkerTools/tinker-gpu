#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"
#include <fstream>

namespace tinker {
/// \addtogroup ff
/// \{

void nData(RcOp);
void massData(RcOp);
void atomicData(RcOp); // atomic number
void xyzData(RcOp);

/// Update #x, #y, #z by #xpos, #ypos, and #zpos.
/// If #xpos etc. are only aliases, return directly.
void copyPosToXyz();

/// Update #x, #y, #z by #xpos, #ypos, and #zpos.
/// If #xpos etc. are only aliases, return directly.
/// \param refreshNBList  If `true`, refresh the neighbor lists at the end.
void copyPosToXyz(bool refreshNBList);

/// Finds the geometric center of each molecule and translate any stray
/// molecules back into the periodic box on GPU.
/// \note
///    - Updating #x, #y, #z is the goal.
///    - Checks whether PBC is in use inside this function.
///    - Will not perturb the neighbor lists so no need to update them.
///    - Tinker uses centers of mass.
void bounds();

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN int padded_n; /// \brief Number of atoms padded by #WARP_SIZE.
TINKER_EXTERN int n;        /// \brief Number of atoms.
TINKER_EXTERN int trajn;    /// \brief Number of the trajectory frames.

TINKER_EXTERN real* x;     ///< Current coordinates used in energy evaluation and neighbor lists.
TINKER_EXTERN real* y;     ///< Current coordinates used in energy evaluation and neighbor lists.
TINKER_EXTERN real* z;     ///< Current coordinates used in energy evaluation and neighbor lists.
TINKER_EXTERN real* trajx; ///< Coordinates of all the trajectory frames.
TINKER_EXTERN real* trajy; ///< Coordinates of all the trajectory frames.
TINKER_EXTERN real* trajz; ///< Coordinates of all the trajectory frames.

/// \var xpos
/// Coordinates used in integrators.
/// \note
///    - New arrays will be allocated only if `sizeof(pos_prec) > sizeof(real)`,
///    otherwise, they will be aliases of #x, #y, and #z.
///    - Whenever #xpos, #ypos, #zpos get updated by the integrator,
///    #x, #y, #z must be updated immediately.
/// \see pos_prec
/// \see real
/// \var ypos
/// \copydoc xpos
/// \var zpos
/// \copydoc xpos
TINKER_EXTERN pos_prec *xpos, *ypos, *zpos;
static_assert(sizeof(pos_prec) >= sizeof(real), "Type pos_prec cannot be shorter than type real.");

TINKER_EXTERN double* mass;    ///< Atomic mass.
TINKER_EXTERN double* massinv; ///< Inversed atomic mass.
TINKER_EXTERN int* atomic;     ///< atomic number.

/// \}
}
