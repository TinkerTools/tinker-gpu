#include "ff/ethrmint.h"
#include "ff/dlmda.h"
#include <algorithm>
#include <tinker/detail/thrmint.hh>
#include <tinker/routines.h>

namespace tinker {
void thermintData(RcOp op)
{
   if (not use_ti)
      return;

   if (op & RcOp::DEALLOC) {
      tilmdahist.clear();
      tilmdadedl.clear();
      tilmdadedlstd.clear();
      tidedllist.clear();
   }

   if (op & RcOp::INIT) {
      tinstepavg = thrmint::tinstepavg;

      // The block accumulators cannot be sized until nstep is known; init_tidyn
      // does that once the window boundaries exist.
      tidedllist.assign(tinstepavg, 0.0);
      tilmdahist.clear();
      tilmdadedl.clear();
      tilmdadedlstd.clear();
      tinbcount = 0;
      tinbtot = 0;
   }
}

void init_tidyn(int nstep)
{
   // The CPU divides the run among the windows, sizes its own accumulators and
   // opens the .ti file; a window left with no dynamics steps is fatal there.
   int ns = nstep;
   tinker_f_inittidyn(&ns);
   tinker_f_prttihead();
   lmdaWinFromFortran();

   tinbtot = thrmint::tinbtot;
   int cap = std::max(1, tinbtot);
   tilmdahist.assign(cap, 0.0);
   tilmdadedl.assign(cap, 0.0);
   tilmdadedlstd.assign(cap, 0.0);
   tidedllist.assign(tinstepavg, 0.0);
   tinbcount = 0;
}

void etidyn(int istep)
{
   // Nothing is stored while the window is equilibrating, or once the schedule
   // has run out.
   int iprod;
   lmdaWinPhase(istep, iprod);
   if (iprod == 0)
      return;
   tidedllist[(iprod - 1) % tinstepavg] = dedl;

   // Reduce a full block into its average and deviation, keeping the lambda
   // that produced it alongside the block itself.
   if (iprod % tinstepavg == 0) {
      double avg, sd;
      avgstd(tidedllist, 0, tinstepavg, avg, sd);
      if (tinbcount < tinbtot) {
         tilmdahist[tinbcount] = lambda;
         tilmdadedl[tinbcount] = avg;
         tilmdadedlstd[tinbcount] = sd;
         ++tinbcount;
      }
   }
}
}
