#include "ff/thermint.h"
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
      tilmdalist.clear();
      tifraclist.clear();
      tiwinend.clear();
      tilmdahist.clear();
      tilmdadedl.clear();
      tilmdadedlstd.clear();
      tidedllist.clear();
   }

   if (op & RcOp::INIT) {
      // Adopt the schedule settisched built.
      tinbin = thrmint::tinbin;
      tinstepavg = thrmint::tinstepavg;
      tieqratio = thrmint::tieqratio;
      tilmda = thrmint::tilmda;
      tilmdalist.assign(thrmint::tilmdalist, thrmint::tilmdalist + tinbin);
      tifraclist.assign(thrmint::tifraclist, thrmint::tifraclist + tinbin);

      // The block accumulators cannot be sized until nstep is known; init_tidyn
      // does that once the window boundaries exist.
      tidedllist.assign(tinstepavg, 0.0);
      tiwinend.clear();
      tilmdahist.clear();
      tilmdadedl.clear();
      tilmdadedlstd.clear();
      tibin = 1;
      tinbcount = 0;
      tinbtot = 0;
      tiwindow = 0;
      tinequil = 0;
      tinblock = 0;
   }
}

void init_tidyn(int nstep)
{
   // The CPU divides the run among the windows, sizes its own accumulators and
   // opens the .ti file; a window left with no dynamics steps is fatal there.
   int ns = nstep;
   tinker_f_inittidyn(&ns);
   tinker_f_prttihead();

   tinbtot = thrmint::tinbtot;
   tibin = thrmint::tibin;
   tilmda = thrmint::tilmda;
   tiwindow = thrmint::tiwindow;
   tinequil = thrmint::tinequil;
   tinblock = thrmint::tinblock;
   tiwinend.assign(thrmint::tiwinend, thrmint::tiwinend + tinbin);

   int cap = std::max(1, tinbtot);
   tilmdahist.assign(cap, 0.0);
   tilmdadedl.assign(cap, 0.0);
   tilmdadedlstd.assign(cap, 0.0);
   tidedllist.assign(tinstepavg, 0.0);
   tinbcount = 0;

   mapSubLambda(tilmda);
}

// Measures the current window against the preceding boundary (thermint.f:298).
static void tiSetWindow()
{
   tiwindow = tiwinend[tibin - 1];
   if (tibin > 1)
      tiwindow = tiwinend[tibin - 1] - tiwinend[tibin - 2];
   tinequil = (int)((double)tiwindow * tieqratio);
   tinblock = (tiwindow - tinequil) / tinstepavg;
}

void etidyn(int istep)
{
   // Nothing is left to sample once the schedule has run out.
   if (tibin > tinbin)
      return;

   int tistart = (tibin > 1) ? tiwinend[tibin - 2] : 0;
   int tistep = istep - tistart;

   // Nothing is stored while the window is equilibrating.
   if (tistep > tinequil) {
      int tiprod = tistep - tinequil;
      tidedllist[(tiprod - 1) % tinstepavg] = dedl;

      // Reduce a full block into its average and deviation, keeping the lambda
      // that produced it alongside the block itself.
      if (tiprod % tinstepavg == 0) {
         double avg, sd;
         avgstd(tidedllist, 0, tinstepavg, avg, sd);
         if (tinbcount < tinbtot) {
            tilmdahist[tinbcount] = tilmda;
            tilmdadedl[tinbcount] = avg;
            tilmdadedlstd[tinbcount] = sd;
            ++tinbcount;
         }
      }
   }

   // Move on to the next lambda window at the window boundary.
   if (istep == tiwinend[tibin - 1])
      tischedule();
}

void tischedule()
{
   // Past the final window the lambda is left where it is and etidyn stops.
   ++tibin;
   if (tibin <= tinbin) {
      tilmda = tilmdalist[tibin - 1];
      tiSetWindow();
      mapSubLambda(tilmda);
   }
}
}
