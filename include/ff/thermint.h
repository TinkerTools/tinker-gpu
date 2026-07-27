#pragma once
#include "tool/rcman.h"
#include <vector>

namespace tinker {
/// True if the "THERM-INTG" keyword is present.
TINKER_EXTERN bool use_ti;
/// Lambda of the current window.
TINKER_EXTERN double tilmda;
/// Fraction of each window discarded as equilibration.
TINKER_EXTERN double tieqratio;
/// Number of lambda windows; the lambda decrement is 1/(tinbin-1).
TINKER_EXTERN int tinbin;
/// Number of steps averaged into one dU/dlambda sample.
TINKER_EXTERN int tinstepavg;
/// Equilibration steps per window, tiwindow*tieqratio.
TINKER_EXTERN int tinequil;
/// Total steps per window, nstep/tinbin.
TINKER_EXTERN int tiwindow;
/// Index of the current window, 0 to tinbin-1.
TINKER_EXTERN int tibin;

/// Block averaged dU/dlambda, one row per lambda window.
TINKER_EXTERN std::vector<std::vector<double>> tilmdadedl;
/// Standard deviation within each block of \ref tilmdadedl.
TINKER_EXTERN std::vector<std::vector<double>> tilmdadedlstd;
/// Per-step dU/dlambda buffer for the block in progress.
TINKER_EXTERN std::vector<double> tidedllist;

/// Reads the TI keywords. Must run before initialize().
void ti_mech();
/// Allocates/initializes the host-side TI accumulators.
void thermintData(RcOp op);
/// Sets up the window geometry once the step count is known.
void init_tidyn(int nstep);
/// Thermodynamic integration driver, one call per MD step.
void etidyn(int istep);
/// Advances to the next lambda window.
void tischedule();
/// Mean and population standard deviation of the first \c count entries.
void avgstd(const std::vector<double>& v, int count, double& avg, double& sd);
}
