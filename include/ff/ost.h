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

/// Evaluates the OST bias at the current lambda and dU/dlambda, with the f term
/// from the mean force with unsampled bins filled, and saves its derivatives
/// for elmdaDyn (eost.f:eostbias).
void eostBias(int vers);

/// The mean force of the lambda bins that the OST flattening bias removes,
/// set by OST-FLATTEN; the free energy estimate always uses \c KERNEL.
enum class OstFlatten
{
   KERNEL, ///< the g kernel cells weighted by exp(g/RT), valid once g is flat in dU/dlambda.
   SAMPLE  ///< the deposits weighted by exp(g/RT) of the g each one saw, valid at any g.
};
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

// the mean force of the flattening bias under OST-FLATTEN SAMPLE: each deposit
// is spread over the lambda bins by its lambda gaussian and weighted by
// exp(g/RT), with g the bicubic g kernel at its center before it was added;
// the sums of a bin are kept relative to the largest g in them, so no weight
// exceeds one, and the shift cancels in the mean.
TINKER_EXTERN OstFlatten ostflatten;            ///< mean force of the flattening bias.
TINKER_EXTERN std::vector<double> ostflatsum;  ///< weighted dU/dlambda sum of each lambda bin.
TINKER_EXTERN std::vector<double> ostflatwt;   ///< total weight of each lambda bin, zero if unsampled.
TINKER_EXTERN std::vector<double> ostflatgmax; ///< largest g of the samples of each lambda bin.
TINKER_EXTERN std::vector<double> ostflatmean; ///< weighted mean dU/dlambda of each lambda bin.

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

// hybrid global + local tempering of the deposited gaussian heights.
TINKER_EXTERN bool use_ostgtemp;      ///< temper heights by the global path bias level.
TINKER_EXTERN bool use_ostltemp;      ///< temper heights by the deposit bin excess over the path level.
TINKER_EXTERN double ostgthresh;      ///< global bias threshold for untempered heights (kcal/mol).
TINKER_EXTERN double ostgtempgamma;   ///< global tempering factor; decay scale is kT*ostgtempgamma.
TINKER_EXTERN double ostlthresh;      ///< local excess threshold for untempered heights (kcal/mol).
TINKER_EXTERN double ostltempgamma;   ///< local tempering factor; decay scale is kT*ostltempgamma.
}
