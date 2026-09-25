#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup polar
/// \{
// electrostatic field due to permanent multipoles
void dfield(real (*field)[3], real (*fieldp)[3]);
void dfieldNonEwald(real (*field)[3], real (*fieldp)[3]);
void dfieldEwald(real (*field)[3], real (*fieldp)[3]);
void dfieldEwaldRecipSelfP1(real (*field)[3]);

// mutual electrostatic field due to induced dipole moments
// -Tu operator
void ufield(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldNonEwald(const real (*uind)[3],
                    const real (*uinp)[3], //
                    real (*field)[3],
                    real (*fieldp)[3]);
void ufieldEwald(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);

void diagPrecond(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3]);

void sparsePrecondBuild();
void sparsePrecondApply(const real (*rsd)[3],
                        const real (*rsdp)[3], //
                        real (*zrsd)[3],
                        real (*zrsdp)[3]);

void ulspredSave(const real (*uind)[3], const real (*uinp)[3]);
void ulspredSum(real (*uind)[3], real (*uinp)[3]);

void inducePrint(const real (*ud)[3]);
void induce(real (*uind)[3], real (*uinp)[3]);

// recovery from unconverged induced dipoles in MD (keyword INDUCE-WIGGLE)
void induceWiggleData(RcOp);
bool induceWiggleOn();   // keyword is set and running MD
bool induceNoPredict();  // true while retrying after a wiggle
void induceReportFailure(const real (*rsd)[3], const real (*rsdp)[3]); // rsdp may be nullptr
bool induceFailed();     // a solver failed in the current energy evaluation
bool induceWiggleRetry(); // wiggles atoms and returns true if energy must be recomputed
/// \}
}

//          | h        | cpp        | acc            | cu
// field    | induce.h | field.cpp  | acc/field*.cpp | field.cu
// pcg      | --       | --         | acc/induce.cpp | pcg.cu
// precond  | induce.h | induce.cpp | acc/induce.cpp | precond.cu
// upredict | induce.h | induce.cpp | acc/induce.cpp | --
