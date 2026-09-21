#include "ff/eabf.h"
#include "ff/energy.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mutant.hh>

#include <vector>

namespace tinker {
void eabfData(RcOp op)
{
   if (not use_abf)
      return;

   if (op & RcOp::DEALLOC) {
      lmdaihist.clear();
      lmdalhist.clear();
      lmdafhist.clear();
      lmdallist.clear();
      lmdaflist.clear();
      lmdafmean.clear();
      lmdafsum.clear();
      lmdafwt.clear();
   }

   if (op & RcOp::INIT) {
      lmdadfdl = 0;

      // Mirror the Fortran allocation/initialization (mutate.f:mutate_abf).
      sizelmdahist = 10000;
      nlmdahist = 0;
      lmdaihist.assign(sizelmdahist + 1, 0);
      lmdalhist.assign(sizelmdahist + 1, 0.0);
      lmdafhist.assign(sizelmdahist + 1, 0.0);
      lmdallist.assign(lmdaintv, 0.0);
      lmdaflist.assign(lmdaintv, 0.0);
      lmdafmean.assign(nlmda + 1, 0.0);
      lmdafsum.assign(nlmda + 1, 0.0);
      lmdafwt.assign(nlmda + 1, 0.0);
   }
}

void eabfBias(int vers)
{
   // The bias depends on lambda alone, so it carries no Cartesian force/virial.
   double eflmda, dfdl;
   efreeLmda(eflmda, dfdl);
   if (vers & calc::energy)
      esum -= eflmda;
   lmdadfdl = dfdl;
}

// abfDeposit -- records the lambda and dU/dlambda average of one accepted
// interval, adds it to the mean force of its lambda bin and updates the free
// energy estimate (eabf.f:abfdeposit).
void abfDeposit(int istep)
{
   // save the interval sample and update the mean force
   nlmdahist = nlmdahist + 1;
   if (nlmdahist > sizelmdahist)
      resizeAbfHist();
   lmdaihist[nlmdahist] = istep;
   lmdalhist[nlmdahist] = lmdaavg;
   lmdafhist[nlmdahist] = dedlavg;
   addAbfHist(nlmdahist);
   lmdadeltag = efreeTot();
}

void addAbfHist(int ihist)
{
   int ilmda = lmdaBin(lmdalhist[ihist]);
   lmdafsum[ilmda] += lmdafhist[ihist];
   lmdafwt[ilmda] += 1.0;
   lmdafmean[ilmda] = lmdafsum[ilmda] / lmdafwt[ilmda];
}

void buildAbfKernel()
{
   for (int i = 1; i <= nlmda; ++i) {
      lmdafmean[i] = 0.0;
      lmdafsum[i] = 0.0;
      lmdafwt[i] = 0.0;
   }
   for (int ihist = 1; ihist <= nlmdahist; ++ihist)
      addAbfHist(ihist);
}

void resizeAbfHist()
{
   // std::vector::resize keeps the existing elements.
   sizelmdahist = 2 * sizelmdahist;
   lmdaihist.resize(sizelmdahist + 1, 0);
   lmdalhist.resize(sizelmdahist + 1, 0.0);
   lmdafhist.resize(sizelmdahist + 1, 0.0);
}

void abfFromFortran()
{
   // initabffile either starts a new history, matching eabfData, or reads an
   // existing one back; the run settings (temperature, mass, friction, dt) are
   // kept by initabffile, so only the history and lambda particle come back.
   sizelmdahist = dlmda::sizelmdahist;
   nlmdahist = dlmda::nlmdahist;
   lmdaihist.assign(sizelmdahist + 1, 0);
   lmdalhist.assign(sizelmdahist + 1, 0.0);
   lmdafhist.assign(sizelmdahist + 1, 0.0);
   for (int k = 1; k <= nlmdahist; ++k) {
      lmdaihist[k] = dlmda::lmdaihist[k - 1];
      lmdalhist[k] = dlmda::lmdalhist[k - 1];
      lmdafhist[k] = dlmda::lmdafhist[k - 1];
   }

   lmdallist.assign(lmdaintv, 0.0);
   lmdaflist.assign(lmdaintv, 0.0);
   lmdafmean.assign(nlmda + 1, 0.0);
   lmdafsum.assign(nlmda + 1, 0.0);
   lmdafwt.assign(nlmda + 1, 0.0);
   for (int i = 1; i <= nlmda; ++i) {
      lmdafmean[i] = dlmda::lmdafmean[i - 1];
      lmdafsum[i] = dlmda::lmdafsum[i - 1];
      lmdafwt[i] = dlmda::lmdafwt[i - 1];
   }

   // energy() maps the sub-lambdas from the main lambda on every call, so the
   // restored lambda takes effect at the first step.
   lmdadeltag = dlmda::lmdadeltag;
   lmdatheta = dlmda::lmdatheta;
   lmdavtheta = dlmda::lmdavtheta;
   lambda = mutant::lambda;
}
}
