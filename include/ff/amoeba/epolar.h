#pragma once
#include "ff/dlmda.h"
#include "ff/precision.h"
#include "ff/termbuf.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup polar
/// \{
void epolarData(RcOp);
void epolar(int vers);
void epolar_adt(int vers);
void epolar_rdt(int vers);
void polarState(RdtMask mask, const int* group);
void epolarEwaldRecipSelf(int vers, AccumRef egvp);
// see also subroutine epolar0e in epolar.f
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3], EnergyBuffer eout);
void epolarPairwiseExtfield(int vers, const real (*uind)[3]);
/// \}
}
