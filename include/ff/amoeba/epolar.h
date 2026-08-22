#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup polar
/// \{

/// Whether the induced dipole dot product supplies the polarization energy for
/// this version. It does whenever an energy is wanted without an analysis
/// breakdown (epolar0e in epolar.f), and then the pairwise and reciprocal
/// kernels must leave every energy channel alone or the two would be summed.
///
/// Under dual topology this also settles the lambda derivative of the energy:
/// the only version that takes its energy pairwise is calc::v3, which
/// lmdaDerivVers() leaves alone, so it carries no derivative channels. The dot
/// product therefore owns dE/dL and d2E/dL2 as well, and the kernels below need
/// no energy derivative sinks.
inline constexpr bool epolarEnergyFromDotProd(int vers)
{
   return (vers & calc::energy) and not(vers & calc::analyz);
}

/// Where one dual topology pass sends its reciprocal-space and self
/// contributions, and with what weight. Polarization masks rpole and polarity
/// before each pass, so a whole pass shares one set of weights and the kernels
/// need no per-atom group lookup.
///
///     E, virial, gradient, torque  <- wa times the pass result
///     their lambda derivatives     <- wb times the same, into the dl sinks
///
/// The default is the ordinary single-topology path: unit weight and no sinks.
/// A null sink means that channel was not requested, so the kernels gate on the
/// pointer rather than on a separate flag.
struct RecipDt
{
   VirialBuffer vdl = nullptr;
   grad_prec* dgx = nullptr;
   grad_prec* dgy = nullptr;
   grad_prec* dgz = nullptr;
   real* dltrqx = nullptr;
   real* dltrqy = nullptr;
   real* dltrqz = nullptr;
   real wa = 1;
   real wb = 0;
};

void epolarData(RcOp);
void epolar(int vers);
void epolar_dt(int vers);
void polarState(RdtMask mask, const int* group);
void epolarEwaldRecipSelf(int vers, EnergyBuffer out_e, VirialBuffer out_v,
   grad_prec* out_gx, grad_prec* out_gy, grad_prec* out_gz);
void epolarEwaldRecipSelfDt(int vers, const RecipDt& dt);
// see also subroutine epolar0e in epolar.f
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3], EnergyBuffer eout);
void epolarPairwiseExtfield(int vers, const real (*uind)[3]);
/// \}
}
