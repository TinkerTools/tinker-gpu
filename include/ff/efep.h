#pragma once
#include "ff/dlmda.h"
#include "tool/rcman.h"
#include <vector>

namespace tinker {
/// Dynamics steps between samples of the trial energies.
TINKER_EXTERN int fepintv;
/// Energy values recorded so far, which doubles as the next free record index.
TINKER_EXTERN int nfep;
/// Total energy values the run can record.
TINKER_EXTERN int nfeptot;

/// Dynamics step at which each energy was recorded.
TINKER_EXTERN std::vector<int> fepstep;
/// Potential energy at the trial lambda of each record.
TINKER_EXTERN std::vector<double> fepene;
/// Main lambda of the simulation for each record.
TINKER_EXTERN std::vector<double> feplmda;
/// Trial main lambda at which each energy was found.
TINKER_EXTERN std::vector<double> feptrial;
/// Periodic box volume of the sample for each record, or 0 without a box.
TINKER_EXTERN std::vector<double> fepvol;

/// Adopts the sampling interval and rewinds the records.
void efepData(RcOp op);
/// Wraps the Fortran initfepdyn and prtfephead, which lay out the lambda
/// windows over the run, count the energies it will record and open the trial
/// energy file.
void init_fepdyn(int nstep);
/// Potential energy of the current structure at each of \c ntrial trial values
/// of the main lambda, and at the native main lambda (efep.f:efeptrial).
///
/// The native lambda is evaluated last and by the same energy-only call as the
/// trial values, so that their differences are those of one set of kernels. On
/// return the main lambda, the sub-lambdas and the induced dipoles are as they
/// were, #esum holds the native energy, and the predictor history and the
/// gradient have not been touched.
void efepTrial(int ntrial, const double* ltrial, double* etrial, double& enative);
/// Whether \c istep is a production step of the current lambda window on which
/// the trial energies are sampled.
bool efepSampleStep(int istep);
/// Finds the potential energy at the main lambda of the current window and at
/// those of its neighbors, at regular intervals over the production portion of
/// the window, and records the values for output along with the box volume
/// (efep.f:efepdyn).
void efepDyn(int istep);
}
