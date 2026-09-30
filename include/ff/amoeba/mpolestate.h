#pragma once
#include "ff/dlmda.h"
#include "ff/pme.h"

namespace tinker {
/// \ingroup mpole
/// \{
/// The shared multipole buffers pole, rpole, cmp and the B-splines. Each one
/// remembers what it was last built from, so the functions below skip
/// rebuilding it from inputs that have not changed. Coordinate and box changes
/// are seen through #xyz_epoch.
///
/// A term starts with mpoleBegin() and then picks what rpole, and with Ewald
/// cmp, are built from with one of the mpoleUse functions. mpoleInit() does
/// both for the common case.

/// Starts a multipole term: zeroes the torque buffers \c vers writes, and with
/// \c zero_vir_trq also #vir_trq; records whether the term computes the virial
/// (mpoleRecipVirial()); and runs chkpole().
void mpoleBegin(int vers, bool zero_vir_trq = true);
/// rpole from pole, and cmp from rpole.
void mpoleUsePole();
/// rpole from poleorig, and cmp from rpole scaled to the electrostatic lambda;
/// with \c build_dl also dlcmp, d cmp / d lambda.
void mpoleUseOrig(bool build_dl);
/// rpole from poleorig, with the sites outside \c mask zeroed, and cmp from it.
void mpoleUseState(RdtMask mask, const int* group);
/// mpoleBegin(), then mpoleUseOrig() with \c do_dlmda or mpoleUsePole()
/// without. dlcmp is built only if \c vers carries a lambda derivative.
void mpoleInit(int vers, bool do_dlmda);
/// rpole from pole, leaving cmp alone, for callers whose cmp already holds the
/// same scaled multipoles.
void mpoleRotateScaled();
/// Scales pole from poleorig to \c lmda, unless it already holds that state.
void mpoleScale(double lmda);
/// Inverts every copy of the multipoles in use at chiral sites whose
/// handedness changed, once per #xyz_epoch.
void chkpole();
/// Brings pole to the electrostatic lambda state, for a term that reads pole
/// itself rather than scaling poleorig on the fly.
void mpoleEnsureElec();
/// Leaves the physical multipoles in rpole: pole at the electrostatic lambda,
/// with charge flux if it is used, rotated at the current coordinates. For
/// readers outside the energy terms, such as the saved dipoles.
void mpoleEnsurePhysical();

// Code that writes the buffers outside the functions above reports it here.

/// Forgets everything recorded, after pole and the local frames are copied in.
void mpoleStateReset();
/// Reports that cmp, fmp or cphi was overwritten in place.
void mpoleCmpClobbered();
/// Reports that alterchg() added the charge flux to the monopoles of pole.
void mpoleFluxApplied();
/// Whether pole still carries the charge flux of the current coordinates.
bool mpoleFluxCurrent();
/// Reports that fmp, fphi and cphi now hold the reciprocal space potential of
/// cmp on the PME grid of this unit, and with \c vir that the convolution first
/// zeroed #vir_m and then accumulated its virial there.
void mpoleFphiProduced(PMEUnit u, bool vir);
/// Whether fmp, fphi and cphi still hold the reciprocal space potential of
/// the current cmp, as this unit would compute it, and with \c need_vir whether
/// #vir_m also still holds the virial of that convolution.
bool mpoleFphiCurrent(PMEUnit u, bool need_vir);
/// Whether the term that last called mpoleBegin() computes the virial, so that
/// its convolutions have to fill #vir_m.
bool mpoleRecipVirial();
/// \}
}
