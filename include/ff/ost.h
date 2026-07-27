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
TINKER_EXTERN double ostlambda;

//====================================================================//
//              OST / metadynamics lambda-dynamics state              //
//====================================================================//

// evaluate the g kernel by bicubic interpolation and fuse the f-kernel update.
TINKER_EXTERN bool ostinterpol;
TINKER_EXTERN bool fastkernel;

// step counters and histogram bookkeeping sizes.
TINKER_EXTERN int iost;         ///< persisted step base (0 unless restarting).
TINKER_EXTERN int iosthist;     ///< steps between histogram deposits.
TINKER_EXTERN int ostnequil;    ///< samples skipped before averaging.
TINKER_EXTERN int ostnavg;      ///< samples averaged per deposit.
TINKER_EXTERN int nlmda;        ///< number of lambda bins.
TINKER_EXTERN int nflmda;       ///< number of dU/dlambda bins.
TINKER_EXTERN int fli0;         ///< bin index where dU/dlambda = 0.
TINKER_EXTERN int nosthist;     ///< number of deposited OST gaussians.
TINKER_EXTERN int sizeosthist;  ///< current OST history allocation.
TINKER_EXTERN int nmetahist;    ///< number of deposited metadynamics gaussians.
TINKER_EXTERN int sizemetahist; ///< current metadynamics history allocation.

// grid widths and gaussian parameters.
TINKER_EXTERN double wlmda;     ///< width of lambda bins.
TINKER_EXTERN double wlmda2;    ///< half width of lambda bins.
TINKER_EXTERN double wflmda;    ///< width of dU/dlambda bins.
TINKER_EXTERN double wflmda2;   ///< half width of dU/dlambda bins.
TINKER_EXTERN double wlhist;    ///< lambda width of new gaussians.
TINKER_EXTERN double wfhist;    ///< dU/dlambda width of new gaussians.
TINKER_EXTERN double maxwlhist; ///< max lambda gaussian width seen.
TINKER_EXTERN double maxwfhist; ///< max dU/dlambda gaussian width seen.
TINKER_EXTERN double hbias;     ///< height of biasing gaussian.
TINKER_EXTERN double oststdev;  ///< gaussian cutoff in standard deviations.
TINKER_EXTERN double osteqratio;///< fraction of interval to equilibrate.

// theta lambda-particle (lambda = sin(theta)^2).
TINKER_EXTERN double osttheta;
TINKER_EXTERN double ostvtheta;
TINKER_EXTERN double ostmass;
TINKER_EXTERN double ostfriction;
TINKER_EXTERN double ostdt;

// current-step derived quantities and running averages.
TINKER_EXTERN double ostdedl;   ///< unbiased dU/dlambda this step.
TINKER_EXTERN double ostdgdl;   ///< dg/dlambda (with chain rule via d2edl2).
TINKER_EXTERN double ostddgdl;  ///< dDeltaG/dlambda this step.
TINKER_EXTERN double deffdl;    ///< effective lambda force for propagation.
TINKER_EXTERN double ostlambdaavg;
TINKER_EXTERN double ostlambdastd;
TINKER_EXTERN double ostlambdaslp; ///< fitted lambda change per sample across the deposit interval.
TINKER_EXTERN double ostdedlavg;
TINKER_EXTERN double ostdedlstd;
TINKER_EXTERN double ostdedlslp;   ///< fitted dU/dlambda change per sample across the deposit interval.
TINKER_EXTERN double eosttot;   ///< current total OST free energy estimate.

TINKER_EXTERN int ostcvbin; ///< number of convergence sub-bins per deposit interval.
TINKER_EXTERN double ostcvdif;
TINKER_EXTERN double ostcvslp;
TINKER_EXTERN double ostcvstd;
TINKER_EXTERN double ostcvrat;

// per-deposit convergence sub-bin averages, size ostcvbin, indexed 0..ostcvbin-1.
TINKER_EXTERN std::vector<double> ostlambdaavgbin;
TINKER_EXTERN std::vector<double> ostlambdastdbin;
TINKER_EXTERN std::vector<double> ostlambdaslpbin;
TINKER_EXTERN std::vector<double> ostdedlavgbin;
TINKER_EXTERN std::vector<double> ostdedlstdbin;
TINKER_EXTERN std::vector<double> ostdedlslpbin;
}
