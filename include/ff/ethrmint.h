#pragma once
#include "ff/dlmda.h"
#include "tool/rcman.h"
#include <vector>

namespace tinker {
/// Number of steps averaged into one dU/dlambda block.
TINKER_EXTERN int tinstepavg;
/// Blocks recorded so far, which doubles as the next free block index.
TINKER_EXTERN int tinbcount;
/// Total blocks the whole schedule can record.
TINKER_EXTERN int tinbtot;

/// Main lambda in effect when each block was recorded.
TINKER_EXTERN std::vector<double> tilmdahist;
/// Block averaged dU/dlambda, in record order.
TINKER_EXTERN std::vector<double> tilmdadedl;
/// Population standard deviation within each block of \ref tilmdadedl.
TINKER_EXTERN std::vector<double> tilmdadedlstd;
/// Per-step dU/dlambda buffer for the block in progress.
TINKER_EXTERN std::vector<double> tidedllist;

/// Adopts the block size and rewinds the accumulators.
void thermintData(RcOp op);
/// Wraps the Fortran inittidyn and prttihead, which lay out the lambda windows
/// over the run and open the block average file.
void init_tidyn(int nstep);
/// Collects dU/dlambda at a dynamics step and averages it into blocks over the
/// production portion of the current lambda window (ethrmint.f:etidyn).
void etidyn(int istep);
}
