#include "ff/thermint.h"
#include "ff/dlmda.h"
#include "ff/ost.h"
#include "tool/argkey.h"
#include "tool/error.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/tinkersuppl.h"
#include <cmath>
#include <cstdio>
#include <tinker/detail/files.hh>

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
      TINKER_THROW("THERM-INTG  --  Requires an alchemical (LAMBDA-DERIV keyword) setup");
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
         avgstd(tidedllist, 0, tinstepavg, avg, sd);
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

// Temporary output path: one row per block average, so the ragged rows of
// tilmdadedl come out as a flat table that numpy.loadtxt reads as-is.
void tiPrint()
{
   if (not use_ti)
      return;

   std::string tifile = FstrView(files::filename)(1, files::leng).trim() + ".ti";
   tifile = tinker_f_version(tifile, "new");
   std::FILE* fp = std::fopen(tifile.c_str(), "w");
   if (fp == nullptr) {
      print(stdout, "\n TI  --  Could not open %s for writing\n", tifile);
      return;
   }

   print(fp, "# tinker9 thermodynamic integration\n");
   print(fp, "# tinbin %d tinstepavg %d tieqratio %.6f tiwindow %d tinequil %d\n", //
      tinbin, tinstepavg, tieqratio, tiwindow, tinequil);
   print(fp, "# window lambda block dedl dedlstd\n");
   for (int w = 0; w < tinbin; ++w) {
      double lam = 1.0 - (double)w / (double)(tinbin - 1);
      if (lam < 0.0)
         lam = 0.0;
      for (size_t b = 0; b < tilmdadedl[w].size(); ++b)
         print(fp, "%6d %12.8f %6zu %20.10e %20.10e\n", //
            w, lam, b, tilmdadedl[w][b], tilmdadedlstd[w][b]);
   }
   std::fclose(fp);

   print(stdout, "\n TI  --  dU/dlambda block averages written to  %s\n", tifile);
}
}
