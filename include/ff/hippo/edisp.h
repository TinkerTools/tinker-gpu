#pragma once
#include "ff/energybuffer.h"
#include "ff/evdw.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup disp
/// \{
bool useDEwald();
/// Whether the tail of a buffered 14-7 vdW is handed to an Ewald dispersion
/// sum whose C6 values are mapped from it (cutoffs.f, kdisp.f:kdsphal).
bool useDHal();
void edispData(RcOp);
void edisp(int vers);

/// Soft core scaling of a mutated damped dispersion pair,
/// \f$ s = \lambda^n (1-S) + \lambda^{n+q} S \f$ with the switch
/// \f$ S = (1-r^2/r_s^2)^3 \f$ inside \f$ r_s \f$ and zero beyond it
/// (edspsc.f), as the powers \c dlsc and \c dlsq. Their lambda derivatives
/// are taken with respect to the main lambda, the van der Waals lambda chain
/// rule included, since \f$ s \f$ is linear in the two powers.
struct DispSoft
{
   real dlsc, dlsq;   ///< \f$ \lambda^n \f$ and \f$ \lambda^{n+q} \f$
   real dlsc1, dlsq1; ///< their first lambda derivatives
   real dlsc2, dlsq2; ///< their second lambda derivatives
   real rs2, rrs2;    ///< the switch radius squared and its reciprocal
};

/// The window of the vdW switch over which an Ewald dispersion tail mapped
/// from a buffered 14-7 vdW takes over from it, the real space pairs weighted
/// by \f$ w = 1 - S(r) \f$, zero up to \c cut and one from \c off on, and
/// whether its sites are damped (edspsc.f:dhalwt).
struct DispHal
{
   real cut, off;
   bool damp;
};
/// The window and damping of useDHal(), or, without it, the full weight from
/// zero on with damping.
DispHal dispHal();

/// The reciprocal and self terms of Ewald dispersion with mutated ligands
/// (edisp.f:dspslot). The C6 of the environment and of each ligand group goes
/// to one of up to three real grids, or slots: groups whose pair weights match
/// share a slot, and a group with no weight left is not placed at all. A pair
/// of sites in slots \c s and \c t is weighted by \c w[s][t], with its lambda
/// derivatives \c dw and \c d2w taken with respect to the main lambda.
struct DispSlots
{
   int nslot;
   int slot[3]; ///< the slot of the environment and of ligands 1 and 2, or -1
   real w[3][3], dw[3][3], d2w[3][3];
};
/// \}
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
TINKER_EXTERN double csixpr;
TINKER_EXTERN real* csix;
TINKER_EXTERN real* adisp;

TINKER_EXTERN int ndspexclude;
TINKER_EXTERN int (*dspexclude)[2];
TINKER_EXTERN real* dspexclude_scale;

TINKER_EXTERN CountBuffer ndisp;
TINKER_EXTERN EnergyBuffer edsp;
TINKER_EXTERN VirialBuffer vir_edsp;
TINKER_EXTERN grad_prec* dedspx;
TINKER_EXTERN grad_prec* dedspy;
TINKER_EXTERN grad_prec* dedspz;
TINKER_EXTERN energy_prec energy_edsp;
TINKER_EXTERN virial_prec virial_edsp[9];

TINKER_EXTERN real dsp2scale;
TINKER_EXTERN real dsp3scale;
TINKER_EXTERN real dsp4scale;
TINKER_EXTERN real dsp5scale;

TINKER_EXTERN double scrdsp;
TINKER_EXTERN double scexpd;
TINKER_EXTERN double scexpdq;
}
