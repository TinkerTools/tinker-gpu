#pragma once
#include "ff/dlmda.h"
#include "ff/precision.h"
#include "tool/rcman.h"
#include <vector>

namespace tinker {
/// Reads the OST/metadynamics engine state from the Fortran modules.
void ost_mech();

/// Allocates/initializes the host-side OST histogram and kernel storage.
void eostData(RcOp op);

/// Evaluates the OST bias at the current lambda and dU/dlambda.
void eostBias(int vers);

/// Orthogonal-space tempering driver (eost.f:eostdyn).
void eostDyn(int istep);

/// One-dimensional lambda metadynamics driver (eost.f:emetadyn).
void eMetaDyn(int istep);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
//====================================================================//
//              OST / metadynamics lambda-dynamics state              //
//====================================================================//

// evaluate the g kernel by bicubic interpolation and fuse the f-kernel update.
TINKER_EXTERN bool ostinterpol;
TINKER_EXTERN bool fastkernel;

// flambda grid and metadynamics history bookkeeping.
TINKER_EXTERN int nflmda;       ///< number of dU/dlambda bins.
TINKER_EXTERN int fli0;         ///< bin index where dU/dlambda = 0.
TINKER_EXTERN int nmetahist;    ///< number of deposited metadynamics gaussians.
TINKER_EXTERN int sizemetahist; ///< current metadynamics history allocation.

// grid widths and gaussian parameters.
TINKER_EXTERN double wflmda;    ///< width of dU/dlambda bins.
TINKER_EXTERN double wflmda2;   ///< half width of dU/dlambda bins.
TINKER_EXTERN double wlhist;    ///< lambda width of new gaussians.
TINKER_EXTERN double wfhist;    ///< dU/dlambda width of new gaussians.
TINKER_EXTERN double maxwlhist; ///< max lambda gaussian width seen.
TINKER_EXTERN double maxwfhist; ///< max dU/dlambda gaussian width seen.
TINKER_EXTERN double hbias;     ///< height of biasing gaussian.
TINKER_EXTERN double oststdev;  ///< gaussian cutoff in standard deviations.

// current-step derived quantities and running averages. The unbiased
// dU/dlambda is \ref dedl itself; OST keeps no separate copy of it.
TINKER_EXTERN double ostdgdl;   ///< dg/dlambda (with chain rule via d2edl2).
TINKER_EXTERN double ostlambdaslp; ///< fitted lambda change per sample across the deposit interval.
TINKER_EXTERN double ostdedlslp;   ///< fitted dU/dlambda change per sample across the deposit interval.

TINKER_EXTERN double ostcvdif;
TINKER_EXTERN double ostcvslp;
TINKER_EXTERN double ostcvstd;
TINKER_EXTERN double ostcvrat;

// hybrid global + local tempering of the deposited gaussian heights.
TINKER_EXTERN bool use_ostgtemp;      ///< temper heights by the global path bias level.
TINKER_EXTERN bool use_ostltemp;      ///< temper heights by the deposit bin excess over the path level.
TINKER_EXTERN double ostgthresh;      ///< global bias threshold for untempered heights (kcal/mol).
TINKER_EXTERN double ostgtempgamma;   ///< global tempering factor; decay scale is kT*ostgtempgamma.
TINKER_EXTERN double ostlthresh;      ///< local excess threshold for untempered heights (kcal/mol).
TINKER_EXTERN double ostltempgamma;   ///< local tempering factor; decay scale is kT*ostltempgamma.
}
