#pragma once
#include "ff/energybuffer.h"
#include "ff/evdw.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup disp
/// \{
bool useDEwald();
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

/// How the reciprocal and self terms of Ewald dispersion weight a mutated
/// ligand by the lambda power of its soft core (edisp.f:dspmutwt4). With
/// \c split, the grid holds the environment C6 in its real part and the ligand
/// C6, the sites \c part flags, in its imaginary part; the convolution
/// separates the two transforms from the copy \c qfwd of the forward
/// transform. The weight derivatives are taken with respect to the main lambda.
struct DispSplit
{
   bool split;
   real wel; ///< ligand-environment weight
   real wll; ///< ligand-ligand weight
   real dwel, dwll;
   real d2wel, d2wll;
   const int* part;
   const real (*qfwd)[2];
   real (*qdl)[2]; ///< the lambda derivative of the potential, for the force one
};
/// \}
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
TINKER_EXTERN real csixpr;
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
