#pragma once
#include "ff/dlmda.h"
#include "tool/rcman.h"
#include <vector>

namespace tinker {
/// Fraction of each window discarded as equilibration.
TINKER_EXTERN double tieqratio;
/// Number of lambda windows.
TINKER_EXTERN int tinbin;
/// Number of steps averaged into one dU/dlambda block.
TINKER_EXTERN int tinstepavg;
/// Equilibration steps in the current window, tiwindow*tieqratio.
TINKER_EXTERN int tinequil;
/// Total dynamics steps in the current window.
TINKER_EXTERN int tiwindow;
/// Number of blocks the current window can hold.
TINKER_EXTERN int tinblock;
/// Number of the current window
TINKER_EXTERN int tibin;
/// Blocks recorded so far, which doubles as the next free block index.
TINKER_EXTERN int tinbcount;
/// Total blocks the whole schedule can record.
TINKER_EXTERN int tinbtot;

/// Main lambda of each window, in schedule order.
TINKER_EXTERN std::vector<double> tilmdalist;
/// Fraction of the run spent in each window; sums to one.
TINKER_EXTERN std::vector<double> tifraclist;
/// Last dynamics step belonging to each window.
TINKER_EXTERN std::vector<int> tiwinend;
/// Main lambda in effect when each block was recorded.
TINKER_EXTERN std::vector<double> tilmdahist;
/// Block averaged dU/dlambda, in record order.
TINKER_EXTERN std::vector<double> tilmdadedl;
/// Population standard deviation within each block of \ref tilmdadedl.
TINKER_EXTERN std::vector<double> tilmdadedlstd;
/// Per-step dU/dlambda buffer for the block in progress.
TINKER_EXTERN std::vector<double> tidedllist;

/// Adopts the precomputed lambda schedule and rewinds the accumulators.
void thermintData(RcOp op);
/// Wraps the Fortran inittidyn and prttihead.
void init_tidyn(int nstep);
/// Thermodynamic integration driver, one call per MD step.
void etidyn(int istep);
/// Advances to the next lambda window.
void tischedule();
}
