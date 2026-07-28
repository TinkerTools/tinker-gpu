#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup polar
/// \{
void epolarData(RcOp);
void epolar(int vers);
void epolar_adt(int vers);
void epolar_rdt(int vers);
void epolar_rdt_staged(int vers);
void polarState(RdtMask mask, const int* group);
void epolarEwaldRecipSelf(int vers, EnergyBuffer out_e, VirialBuffer out_v,
   grad_prec* out_gx, grad_prec* out_gy, grad_prec* out_gz);
// see also subroutine epolar0e in epolar.f
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3], EnergyBuffer eout);
void epolarPairwiseExtfield(int vers, const real (*uind)[3]);
/// \}
}
