#pragma once
#include "tool/rcman.h"

namespace tinker {
/// \ingroup mplar
bool useEmplar();

/// \ingroup mplar
void emplarData(RcOp);

/// \ingroup mplar
/// \brief Multipole and AMOEBA polarization energy.
/// \note Will not be called in any of the following situations:
///    - not using GPU;
///    - not using CUDA as the primary GPU package;
///    - not using periodic boundary condition;
///    - not using both multipole and AMOEBA polarization terms.
/// \note Does not count number of interactions and aborts the program
/// if called erroneously (bug in the code).
void emplar(int vers);

/// \ingroup mplar
void emplar_adt(int vers);

/// \ingroup mplar
void emplar_rdt(int vers);

/// \ingroup mplar
void emplar_rdt_staged(int vers);
}
