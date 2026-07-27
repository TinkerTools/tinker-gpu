#include "ff/thermint.h"
#include "ff/dlmda.h"
#include "ff/ost.h"
#include "tool/argkey.h"
#include "tool/error.h"
#include <cmath>

namespace tinker {
void ti_mech()
{
   getKV("THERM-INTG", use_ti, false);
   getKV("TI-NBIN", tinbin, 21);
   getKV("TI-NSTEPAVG", tinstepavg, 100);
   getKV("TI-EQUIL-RATIO", tieqratio, 0.5);

   if (not use_ti)
      return;

   if (use_ost or use_meta)
      TINKER_THROW("THERM-INTG  --  Not compatible with OST or metadynamics");
   if (not use_dlmda)
      TINKER_THROW("THERM-INTG  --  Requires an alchemical (MUTATE) setup");
   if (tinbin < 2)
      TINKER_THROW("THERM-INTG  --  TI-NBIN must be at least 2");
   if (tinstepavg < 1)
      TINKER_THROW("THERM-INTG  --  TI-NSTEPAVG must be positive");
   if (tieqratio < 0.0 or tieqratio >= 1.0)
      TINKER_THROW("THERM-INTG  --  TI-EQUIL-RATIO must be in [0,1)");
}

void thermintData(RcOp op)
{
   if (not use_ti)
      return;

   if (op & RcOp::DEALLOC) {
      tilmdadedl.clear();
      tilmdadedl.shrink_to_fit();
      tilmdadedlstd.clear();
      tilmdadedlstd.shrink_to_fit();
      tidedllist.clear();
      tidedllist.shrink_to_fit();
   }

   if (op & RcOp::INIT) {
      // One row per lambda window; the rows grow by push_back.
      tilmdadedl.assign(tinbin, {});
      tilmdadedlstd.assign(tinbin, {});
      tidedllist.assign(tinstepavg, 0.0);
      tibin = 0;
      tilmda = 1.0;
      tiwindow = 0;
      tinequil = 0;
   }
}

void init_tidyn(int nstep)
{
   tiwindow = nstep / tinbin;
   if (tiwindow < 1)
      TINKER_THROW("THERM-INTG  --  Fewer dynamics steps than TI-NBIN windows");

   tinequil = (int)(tiwindow * tieqratio);
   if (tiwindow - tinequil < tinstepavg)
      TINKER_THROW("THERM-INTG  --  Production block is shorter than TI-NSTEPAVG");

   tibin = 0;
   tilmda = 1.0;
   mapSubLambda(tilmda);
}

void avgstd(const std::vector<double>& v, int count, double& avg, double& sd)
{
   if (count < 1) {
      avg = 0.0;
      sd = 0.0;
      return;
   }

   // Shifted mean, as in histstat, to keep the sum of squares well conditioned.
   double k = v[0];
   double total = 0.0, totalsq = 0.0;
   for (int i = 0; i < count; ++i) {
      double d = v[i] - k;
      total += d;
      totalsq += d * d;
   }
   avg = k + total / (double)count;
   double var = (totalsq - total * total / (double)count) / (double)count;
   sd = std::sqrt(var > 0.0 ? var : 0.0);
}

void etidyn(int istep)
{
   // A trailing partial window is left unsampled when nstep % tinbin != 0.
   if (tibin >= tinbin)
      return;

   int tistep = (istep - 1) % tiwindow + 1;

   // Nothing is stored while the window is equilibrating.
   if (tistep > tinequil) {
      int tiprod = tistep - tinequil;
      tidedllist[(tiprod - 1) % tinstepavg] = dedl;
      if (tiprod % tinstepavg == 0) {
         double avg, sd;
         avgstd(tidedllist, tinstepavg, avg, sd);
         tilmdadedl[tibin].push_back(avg);
         tilmdadedlstd[tibin].push_back(sd);
      }
   }

   if (tistep == tiwindow)
      tischedule();
}

void tischedule()
{
   // Both endpoints are sampled, so tinbin windows span [0,1] in tinbin-1 steps.
   tibin = tibin + 1;
   tilmda = 1.0 - (double)tibin / (double)(tinbin - 1);
   if (tilmda < 0.0)
      tilmda = 0.0;
   // energy() re-maps the sub-lambdas at the top of the next step.
}
}
