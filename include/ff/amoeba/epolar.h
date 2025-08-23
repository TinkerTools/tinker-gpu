#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup polar
/// \{
void epolarData(RcOp);
void epolar(int vers);
void epolarEwaldRecipSelf(int vers);
// see also subroutine epolar0e in epolar.f
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3]);
void epolarPairwiseExtfield(int vers, const real (*uind)[3]);
/// \}
}
