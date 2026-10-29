#include "ff/efep.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include <algorithm>
#include <tinker/detail/fep.hh>
#include <tinker/routines.h>

namespace tinker {
void efepData(RcOp op)
{
   if (not use_fep)
      return;

   if (op & RcOp::DEALLOC) {
      fepstep.clear();
      fepene.clear();
      feplmda.clear();
      feptrial.clear();
      fepvol.clear();
   }

   if (op & RcOp::INIT) {
      fepintv = fep::fepintv;

      // The records cannot be sized until nstep is known; init_fepdyn does
      // that once the window boundaries exist.
      fepstep.clear();
      fepene.clear();
      feplmda.clear();
      feptrial.clear();
      fepvol.clear();
      nfep = 0;
      nfeptot = 0;
   }
}

void init_fepdyn(int nstep)
{
   // The CPU divides the run among the windows, counts the energies they will
   // record, sizes its own arrays and opens the .fep file; a window left with
   // no dynamics steps is fatal there.
   int ns = nstep;
   tinker_f_initfepdyn(&ns);
   tinker_f_prtfephead();
   lmdaWinFromFortran();

   nfeptot = fep::nfeptot;
   int cap = std::max(1, nfeptot);
   fepstep.assign(cap, 0);
   fepene.assign(cap, 0.0);
   feplmda.assign(cap, 0.0);
   feptrial.assign(cap, 0.0);
   fepvol.assign(cap, 0.0);
   nfep = 0;
}

void efepTrial(int ntrial, const double* ltrial, double* etrial, double& enative)
{
   // A barostat that rescales the coordinates after the force call of the step
   // leaves the neighbor lists behind the structure; this is a no-op when the
   // atoms have not moved.
   nblistRefresh();

   // Save the native main lambda. The native induced dipoles come back when
   // the trial ends; its solves start from the direct guess and are not saved,
   // so the predictor history stays that of the dynamics.
   const double lnative = lambda;
   InducedTrial trial(InducedTrial::History::BYPASS);

   // get the potential energy at each of the trial lambda values
   for (int i = 0; i < ntrial; ++i) {
      lambda = ltrial[i];
      energy(calc::energy);
      etrial[i] = esum;
   }

   // Get the native energy last, which maps the native sub-lambdas and
   // scales the lambda dependent parameters to them again.
   lambda = lnative;
   energy(calc::energy);
   enative = esum;
}

bool efepSampleStep(int istep)
{
   int iprod;
   lmdaWinPhase(istep, iprod);
   return iprod != 0 and iprod % fepintv == 0;
}

void efepDyn(int istep)
{
   // Nothing is found while the window is equilibrating, once the schedule has
   // run out, or between the sampling intervals.
   if (not efepSampleStep(istep))
      return;

   // the trial values are those of the windows before and after
   int ntrial = 0;
   double ltrial[2], etrial[2];
   if (lmdawin > 1)
      ltrial[ntrial++] = lmdawinlist[lmdawin - 2];
   if (lmdawin < nlmdawin)
      ltrial[ntrial++] = lmdawinlist[lmdawin];
   if (nfep + ntrial + 1 > nfeptot)
      return;

   // get the energy at the native and at the trial lambda values
   double enative;
   efepTrial(ntrial, ltrial, etrial, enative);

   // all energies of the sample share the current box volume
   double vol = 0;
   if (box_shape != BoxShape::UNBOUND)
      vol = boxVolume();

   // record the native energy, then that of each trial lambda
   fepstep[nfep] = istep;
   fepene[nfep] = enative;
   feplmda[nfep] = lambda;
   feptrial[nfep] = lambda;
   fepvol[nfep] = vol;
   ++nfep;
   for (int i = 0; i < ntrial; ++i) {
      fepstep[nfep] = istep;
      fepene[nfep] = etrial[i];
      feplmda[nfep] = lambda;
      feptrial[nfep] = ltrial[i];
      fepvol[nfep] = vol;
      ++nfep;
   }
}
}
