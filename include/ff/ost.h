#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// Mapping type from the OST main lambda to a sub-lambda.
enum class Ostmap
{
   NONE, ///< quintic taper
   EXP,  ///< power law
   INV,  ///< shifted inverse power
   QNT   ///< quantized (taper plus endpoint derivative flags)
};

/// Parses a Fortran character*3 map selector into an Ostmap value.
Ostmap ostmapFrom(const char* s);

void mapSubLambda();

void lmdachain(int vers);

/// Allocates/initializes the host-side OST histogram and kernel storage.
void eostData(RcOp op);

/// Orthogonal-space tempering driver (eost.f:eostdyn).
void eostDyn(int istep, int vers);

/// One-dimensional lambda metadynamics driver (eost.f:emetadyn).
void eMetaDyn(int istep);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
// exponential mapping exponents.
TINKER_EXTERN int ostemexp;
TINKER_EXTERN int ostepexp;
TINKER_EXTERN int ostevexp;

// inverse-power mapping exponents and shifts.
TINKER_EXTERN int ostinvemn;
TINKER_EXTERN int ostinvepn;
TINKER_EXTERN int ostinvevn;
TINKER_EXTERN double ostinvemeps;
TINKER_EXTERN double ostinvepeps;
TINKER_EXTERN double ostinveveps;

// mapping-type selectors.
TINKER_EXTERN Ostmap ostemap;
TINKER_EXTERN Ostmap ostpmap;
TINKER_EXTERN Ostmap ostvmap;

// sub-lambda bounds used by the taper/quantized maps.
TINKER_EXTERN double ostelmda0;
TINKER_EXTERN double ostelmda1;
TINKER_EXTERN double ostplmda0;
TINKER_EXTERN double ostplmda1;
TINKER_EXTERN double ostvlmda0;
TINKER_EXTERN double ostvlmda1;

// main lambda value and OST/metadynamics flags.
TINKER_EXTERN double ostlambda;
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_pol4i;
TINKER_EXTERN bool use_pol4f;

// first and second derivatives of each sub-lambda w.r.t. the main lambda.
TINKER_EXTERN double deldlmda;
TINKER_EXTERN double dpldlmda;
TINKER_EXTERN double dvldlmda;
TINKER_EXTERN double d2eldlmda2;
TINKER_EXTERN double d2pldlmda2;
TINKER_EXTERN double d2vldlmda2;

//====================================================================//
//              OST / metadynamics lambda-dynamics state              //
//====================================================================//

// dynamics-mode flags (mirror ost::use_ostdyn / use_metadyn).
TINKER_EXTERN bool use_ostdyn;
TINKER_EXTERN bool use_metadyn;
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
TINKER_EXTERN double ostdedlavg;
TINKER_EXTERN double ostdedlstd;
TINKER_EXTERN double eosttot;   ///< current total OST free energy estimate.

/// Forces the energy term on when OST/metadynamics is active.
inline int ostVers(int vers)
{
   return (use_ost or use_meta) ? (vers | calc::energy) : vers;
}
}
