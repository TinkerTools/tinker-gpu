#include "ff/eabf.h"
#include "ff/energy.h"
#include "ff/eost.h"
#include "math/const.h"
#include "md/misc.h"
#include "test.h"
#include "testrt.h"
#include "tinker9.h"
#include "tool/iofortstr.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/files.hh>
#include <tinker/routines.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// Port of the Fortran ABF unit tests of the ABF routines in
// tinker/test/test_eabf.f. Pure host math -- no molecular system, no GPU.
//
// The expected values and tolerances are copied from test_eabf.f. A failing
// check indicates a real discrepancy in src/eabf.cpp, to be fixed there -- do
// not adjust the expected values.

using namespace tinker;

namespace {
// resetabf -- set the lambda bias state shared with OST to unit-test defaults
// and leave every OST-only array empty, so ABF must never touch them
// (test_eabf.f:resetabf).
void resetabf(int nl, int nhist)
{
   nlmda = nl;
   wlmda = 1.0 / (double)(nlmda - 1);
   wlmda2 = 0.5 * wlmda;
   nlmdahist = 0;
   sizelmdahist = nhist;
   lmdastep = 0;
   lmdaintv = 10;
   lmdanpa = 3;
   lmdanpb = 3;
   lmdanpc = 4;
   lambda = 0.0;
   lmdaavg = 0.0;
   lmdastd = 0.0;
   dedl = 0.0;
   dedlavg = 0.0;
   dedlstd = 0.0;
   deffdl = 0.0;
   lmdaddgdl = 0.0;
   lmdadeltag = 0.0;
   lmdadfdl = 0.0;
   lmdadt = 0.0;
   use_abf = true;
   use_ost = false;
   use_meta = false;
   use_lmdacv = false;
   lmdacvstd = 0.0;
   lmdacvrat = 0.0;
   use_lmdafill = true;
   lmdafillr = 0.1;

   lmdaihist.assign(sizelmdahist + 1, 0);
   lmdalhist.assign(sizelmdahist + 1, 0.0);
   lmdafhist.assign(sizelmdahist + 1, 0.0);
   lmdallist.assign(lmdaintv, 0.0);
   lmdaflist.assign(lmdaintv, 0.0);
   lmdafmean.assign(nlmda + 1, 0.0);
   lmdafsum.assign(nlmda + 1, 0.0);
   lmdafwt.assign(nlmda + 1, 0.0);

   osthist.clear();
   ostnext.clear();
   osthead.clear();
   osthhist.clear();
   ostwlhist.clear();
   ostwfhist.clear();
   gkernel.clear();
   gfkernel.clear();
   glfkernel.clear();
   glkernel.clear();
   vkernelmax.clear();
}

// fills the first nbin bins of the sampled mean force fsamp with the bins
// flagged in sampled, within nreach bins of data, checks the result against ref
// to within eps and against the Fortran lmdafill exactly, and checks that fsamp
// is left unchanged
void checkFill(int nbin, const std::vector<double>& fsamp, const std::vector<bool>& sampled, int nreach,
   const std::vector<double>& ref, double eps)
{
   // the engine arrays are indexed 1..nbin, the Fortran ones 0..nbin-1
   std::vector<double> f1(nbin + 1, 0.0), fbias(nbin + 1, 0.0);
   std::vector<bool> s1(nbin + 1, false);
   std::vector<int> sf(nbin);
   std::vector<double> ff(fsamp.begin(), fsamp.begin() + nbin), fbiasf(nbin, 0.0);
   for (int i = 1; i <= nbin; ++i) {
      f1[i] = fsamp[i - 1];
      s1[i] = sampled[i - 1];
      sf[i - 1] = sampled[i - 1] ? 1 : 0;
   }
   const std::vector<double> fsave = f1;
   lmdaFill(nbin, f1, s1, nreach, fbias);
   tinker_f_lmdafill(&nbin, ff.data(), sf.data(), &nreach, fbiasf.data());
   for (int i = 1; i <= nbin; ++i) {
      CAPTURE(i);
      COMPARE_REALS(fbias[i], ref[i - 1], eps);
      REQUIRE(fbias[i] == fbiasf[i - 1]);
      REQUIRE(f1[i] == fsave[i]);
   }
}
}

TEST_CASE("EABF-bias", "[ff][eabf][L2]")
{
   // seed a linear mean force f(lambda) = 2 + 4*lambda on the grid, with data
   // in every bin so none is filled
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   for (int i = 1; i <= nlmda; ++i)
      testSetLmdaBin(i, 2.0 + 4.0 * (double)(i - 1) * wlmda);
   for (int k = 0; k < 9; ++k)
      vir[k] = 1.0;
   lambda = 0.3;
   esum = 1.0;
   eabfBias(calc::energy);

   // the free energy 2*lambda + 2*lambda**2 leaves the energy and its slope
   // 2 + 4*lambda is saved for the lambda particle
   COMPARE_REALS(esum, 1.0 - 0.78, 1.0e-12);
   COMPARE_REALS(lmdadfdl, 3.2, 1.0e-12);
   for (int k = 0; k < 9; ++k)
      REQUIRE(vir[k] == 1.0);
}

TEST_CASE("EABF-dyn", "[ff][eabf][L2]")
{
   // a settled interval records its lambda and dU/dlambda average
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   lmdaintv = 4;
   lmdanpa = 0;
   lmdanpb = 0;
   lmdanpc = 4;
   lmdadt = 0.0;
   lmdadfdl = 0.5;
   for (int istep = 1; istep <= lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 3.0;
      elmdaDyn();
      if (istep < lmdaintv)
         COMPARE_INTS(nlmdahist, 0); // waits for the interval end
   }
   COMPARE_REALS(deffdl, 2.5, 1.0e-12);
   COMPARE_INTS(nlmdahist, 1);
   COMPARE_INTS(lmdaihist[1], lmdaintv);
   REQUIRE(lmdalhist[1] == 0.5);
   COMPARE_REALS(lmdafhist[1], 3.0, 1.0e-12);
   REQUIRE(lmdafwt[3] == 1.0);
   COMPARE_REALS(lmdafmean[3], 3.0, 1.0e-12);
   COMPARE_REALS(lmdadeltag, 0.75, 1.0e-12);

   // a noisy interval is still recorded while the gate is off
   for (int istep = lmdaintv + 1; istep <= 2 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 1.0;
      if (istep % 2 == 0)
         dedl = 11.0;
      elmdaDyn();
   }
   COMPARE_INTS(nlmdahist, 2);
   REQUIRE(lmdafwt[3] == 2.0);
   COMPARE_REALS(lmdafmean[3], 4.5, 1.0e-12);

   // with the gate on a noisy interval leaves the history unchanged
   use_lmdacv = true;
   lmdacvstd = 1.0;
   lmdacvrat = 0.0;
   double egsave = lmdadeltag;
   for (int istep = 2 * lmdaintv + 1; istep <= 3 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 1.0;
      if (istep % 2 == 0)
         dedl = 11.0;
      elmdaDyn();
   }
   COMPARE_INTS(nlmdahist, 2);
   REQUIRE(lmdafwt[3] == 2.0);
   COMPARE_REALS(lmdafmean[3], 4.5, 1.0e-12);
   COMPARE_REALS(lmdadeltag, egsave, 1.0e-12);

   // with the gate on a settled interval is still recorded
   for (int istep = 3 * lmdaintv + 1; istep <= 4 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 3.0;
      elmdaDyn();
   }
   COMPARE_INTS(nlmdahist, 3);
   REQUIRE(lmdafwt[3] == 3.0);
   COMPARE_REALS(lmdafmean[3], 4.0, 1.0e-12);
}

TEST_CASE("EABF-mean", "[ff][eabf][L2]")
{
   // the bias force follows the running mean of the bin samples
   TestLmdaFlagGuard guard;
   resetabf(5, 8);
   for (int i = 1; i <= 6; ++i) {
      lmdalhist[i] = 0.25;
      lmdafhist[i] = (double)i;
      nlmdahist = i;
      addAbfHist(i);
      COMPARE_REALS(lmdafmean[2], 0.5 * (double)(i + 1), 1.0e-12);
   }

   // a rebuild from the history reproduces the accumulated bins
   buildAbfKernel();
   REQUIRE(lmdafwt[2] == 6.0);
   COMPARE_REALS(lmdafsum[2], 21.0, 1.0e-12);
   COMPARE_REALS(lmdafmean[2], 3.5, 1.0e-12);
   REQUIRE(lmdafmean[1] == 0.0);
}

TEST_CASE("EABF-gate", "[ff][eabf][L2]")
{
   // drive an interval with a deterministic frictionless lambda particle, so
   // that any lambda motion comes from the gate alone
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   lmdaintv = 6;
   lmdanpa = 2;
   lmdanpb = 2;
   lmdanpc = 2;
   lmdadt = 0.1;
   lmdamass = 1.0;
   lmdafric = 0.0;
   lmdatheta = 0.25 * pi;
   lmdavtheta = 0.0;
   lambda = 0.5;
   lmdadfdl = 0.0;

   // lam is indexed by istep
   double lam[7] = {0};
   for (int istep = 1; istep <= lmdaintv; ++istep) {
      dedl = 1.0;
      elmdaDyn();
      lam[istep] = lambda;
   }

   // the particle moves only while the interval is in phase a
   REQUIRE(lam[1] != lam[2]);
   double frozen = lam[2];
   for (int istep = 3; istep <= lmdaintv; ++istep)
      REQUIRE(lam[istep] == frozen);

   // the sample is recorded exactly on the frozen lambda
   COMPARE_INTS(nlmdahist, 1);
   REQUIRE(lmdalhist[1] == frozen);
   COMPARE_REALS(lmdafhist[1], 1.0, 1.0e-12);
}

TEST_CASE("EABF-resize", "[ff][eabf][L2]")
{
   // three samples overflow a history sized for two
   TestLmdaFlagGuard guard;
   resetabf(5, 2);
   lmdaintv = 4;
   lmdanpa = 0;
   lmdanpb = 0;
   lmdanpc = 4;
   lmdadt = 0.0;
   lmdadfdl = 0.0;
   for (int istep = 1; istep <= 3 * lmdaintv; ++istep) {
      lambda = 0.25;
      dedl = (double)((istep - 1) / lmdaintv + 1);
      elmdaDyn();
   }
   COMPARE_INTS(nlmdahist, 3);
   COMPARE_INTS(sizelmdahist, 4);
   COMPARE_INTS(lmdaihist[1], 4);
   COMPARE_INTS(lmdaihist[3], 12);
   COMPARE_REALS(lmdafhist[1], 1.0, 1.0e-12);
   COMPARE_REALS(lmdafhist[2], 2.0, 1.0e-12);
   COMPARE_REALS(lmdafhist[3], 3.0, 1.0e-12);
   REQUIRE(lmdalhist[3] == 0.25);
   COMPARE_REALS(lmdafmean[2], 2.0, 1.0e-12);
}

TEST_CASE("EABF-fill", "[ff][eabf][L2]")
{
   // A copy of the nearest sampled bin past either end of the data, linear
   // interpolation inside a gap, zero with no data, sampled bins kept exactly,
   // and the sampled values left unchanged, then the reach: a gap up to twice
   // the reach is interpolated, a wider gap or an end is copied within reach
   // and zero beyond it (test_eabf.f:test_eabf_fill).
   const bool T = true, F = false;

   // bins below the first sampled bin copy its mean force
   checkFill(5, {0., 0., 0., -20., -40.}, {F, F, F, T, T}, 5, {-20., -20., -20., -20., -40.}, 0.0);

   // bins past the last sampled bin copy its mean force
   checkFill(5, {-10., -20., 0., 0., 0.}, {T, T, F, F, F}, 5, {-10., -20., -20., -20., -20.}, 0.0);

   // a gap between sampled bins is interpolated linearly
   checkFill(5, {-10., 0., 0., -40., -50.}, {T, F, F, T, T}, 5, {-10., -20., -30., -40., -50.}, 1.0e-12);

   // a bin halfway between two sampled bins takes their average
   checkFill(3, {-10., 0., -30.}, {T, F, T}, 3, {-10., -20., -30.}, 0.0);

   // both ends and a gap are filled in the same profile
   checkFill(5, {0., -10., 0., 0., -40.}, {F, T, F, F, T}, 5, {-10., -10., -20., -30., -40.}, 1.0e-12);

   // a sampled zero is data, and values in unsampled bins are unused
   checkFill(5, {7., 9., 0., -20., -40.}, {F, F, T, T, T}, 5, {0., 0., 0., -20., -40.}, 0.0);

   // with no sampled bin the profile is zero
   checkFill(5, {3., 1., 4., 1., 5.}, {F, F, F, F, F}, 5, {0., 0., 0., 0., 0.}, 0.0);

   // a fully sampled profile is reproduced exactly
   checkFill(5, {1.1, -2.3, 3.7, 0., -5.9}, {T, T, T, T, T}, 5, {1.1, -2.3, 3.7, 0., -5.9}, 0.0);

   // a single bin keeps its data, or is zero without it
   checkFill(1, {2.5}, {T}, 1, {2.5}, 0.0);
   checkFill(1, {2.5}, {F}, 1, {0.}, 0.0);

   // a gap of twice the reach is interpolated across; unsampled bins hold junk
   // values here, which must never reach the fill
   checkFill(3, {-10., 7., -30.}, {T, F, T}, 1, {-10., -20., -30.}, 1.0e-12);
   checkFill(5, {-10., 7., 9., 11., -50.}, {T, F, F, F, T}, 2, {-10., -20., -30., -40., -50.}, 1.0e-12);

   // a gap wider than twice the reach copies each side within reach
   checkFill(4, {-10., 7., 9., -40.}, {T, F, F, T}, 1, {-10., -10., -40., -40.}, 0.0);
   checkFill(5, {-10., 7., 9., 11., -50.}, {T, F, F, F, T}, 1, {-10., -10., 0., -50., -50.}, 0.0);

   // past either end of the data a bin within reach copies it
   checkFill(5, {7., 9., 11., -40., 13.}, {F, F, F, T, F}, 1, {0., 0., -40., -40., -40.}, 0.0);

   // a reach of zero fills nothing
   checkFill(5, {7., -10., 9., -40., 11.}, {F, T, F, T, F}, 0, {0., -10., 0., -40., 0.}, 0.0);
}

TEST_CASE("EABF-fillbias", "[ff][eabf][L2]")
{
   // The ABF bias and the slope saved for the lambda particle follow the
   // filled mean force, while the free energy estimate and the bin mean force
   // keep the sampled data only; a fully sampled profile gives the same bias
   // with or without the fill. resetabf frees the OST kernels, so this also
   // shows the ABF fill never reads an OST array (test_eabf.f:test_eabf_fillbias).
   TestLmdaFlagGuard guard;
   resetabf(5, 8);

   // sample the two top bins only, at lambda 0.75 and 1, with a reach that
   // fills every bin
   lmdafillr = 1.0;
   nlmdahist = 2;
   lmdalhist[1] = 0.75;
   lmdafhist[1] = -20.0;
   lmdalhist[2] = 1.0;
   lmdafhist[2] = -40.0;
   for (int i = 1; i <= nlmdahist; ++i)
      addAbfHist(i);

   // the filled bias holds a mean force of -20 below lambda 0.75
   use_lmdafill = true;
   lambda = 1.0;
   esum = 0.0;
   eabfBias(calc::energy);
   COMPARE_REALS(-bostlmda, 22.5, 1.0e-12);
   COMPARE_REALS(esum, 22.5, 1.0e-12);
   REQUIRE(lmdadfdl == -40.0);
   lambda = 0.3;
   esum = 0.0;
   eabfBias(calc::energy);
   COMPARE_REALS(-bostlmda, 6.0, 1.0e-12);
   REQUIRE(lmdadfdl == -20.0);

   // the bias that drives lambda is the one eabfBias removes
   double vbias, slope;
   efreeBias(lambda, vbias, slope);
   REQUIRE(vbias == bostlmda);
   REQUIRE(slope == lmdadfdl);

   // the estimate keeps zero mean force in the unsampled bins
   COMPARE_REALS(efreeTot(), -10.0, 1.0e-12);
   REQUIRE(lmdafmean[1] == 0.0);
   REQUIRE(lmdafmean[4] == -20.0);

   // a reach of one bin fills only the bin below lambda 0.75, so the mean
   // force is 0, 0, -20, -20, -40
   lmdafillr = 0.25;
   lambda = 0.3;
   eabfBias(calc::energy);
   COMPARE_REALS(-bostlmda, 0.1, 1.0e-12);
   COMPARE_REALS(lmdadfdl, -4.0, 1.0e-12);
   lambda = 0.2;
   eabfBias(calc::energy);
   REQUIRE(bostlmda == 0.0);
   REQUIRE(lmdadfdl == 0.0);
   lmdafillr = 1.0;

   // without the fill the bias is the sampled free energy
   use_lmdafill = false;
   lambda = 0.3;
   eabfBias(calc::energy);
   REQUIRE(bostlmda == 0.0);
   REQUIRE(lmdadfdl == 0.0);
   lambda = 0.6;
   eabfBias(calc::energy);
   COMPARE_REALS(-bostlmda, 0.4, 1.0e-12);
   COMPARE_REALS(lmdadfdl, -8.0, 1.0e-12);

   // once every bin is sampled the fill changes nothing
   nlmdahist = 5;
   lmdalhist[3] = 0.0;
   lmdafhist[3] = 2.0;
   lmdalhist[4] = 0.25;
   lmdafhist[4] = 3.0;
   lmdalhist[5] = 0.5;
   lmdafhist[5] = -5.0;
   for (int i = 3; i <= nlmdahist; ++i)
      addAbfHist(i);
   lambda = 0.6;
   use_lmdafill = true;
   eabfBias(calc::energy);
   const double vfill = bostlmda, dfill = lmdadfdl;
   use_lmdafill = false;
   eabfBias(calc::energy);
   REQUIRE(vfill == bostlmda);
   REQUIRE(dfill == lmdadfdl);
}

TEST_CASE("EABF-fillfull", "[ff][eabf][L2]")
{
   // On the default grid of 201 bins a fill reach of one lambda, or more,
   // gives the unlimited fill, where each end copies the nearest sampled bin
   // and a gap of any width is interpolated, while a reach of 0.25, or the
   // default 0.1, copies each side of a wider gap and the low end only within
   // reach; each case is compared with a profile built here by hand
   // (test_eabf.f:test_eabf_fillfull).
   TestLmdaFlagGuard guard;

   // sample only lambda 0.3 and 0.9, leaving 60 bins below the data, a gap of
   // 120 bins and 20 bins above the data
   resetabf(201, 1);
   use_lmdafill = true;
   const int ilo = 61, ihi = 181;
   const double flo = -20.0, fhi = -80.0;
   testSetLmdaBin(ilo, flo);
   testSetLmdaBin(ihi, fhi);

   // the reach and the bins it covers, where a reach of one or more covers
   // every bin; 0.25 is 50 bins and 0.1 is 20, so the gap is wider than twice
   // either reach
   const int nall = nlmda;
   const struct
   {
      double reach;
      int nr;
   } cases[] = {{1.0, nall}, {1.5, nall}, {0.25, 50}, {0.1, 20}};
   for (const auto& c : cases) {
      CAPTURE(c.reach);
      lmdafillr = c.reach;

      // build the expected profile by hand
      std::vector<double> fref(nlmda + 1, 0.0);
      for (int k = 1; k <= nlmda; ++k) {
         if (c.nr == nall) {
            if (k <= ilo)
               fref[k] = flo;
            else if (k >= ihi)
               fref[k] = fhi;
            else
               fref[k] = flo + (double)(k - ilo) / (double)(ihi - ilo) * (fhi - flo);
         } else {
            if (k <= ilo) {
               if (ilo - k <= c.nr)
                  fref[k] = flo;
            } else if (k < ihi) {
               if (k - ilo <= c.nr)
                  fref[k] = flo;
               if (ihi - k <= c.nr)
                  fref[k] = fhi;
            } else {
               if (k - ihi <= c.nr)
                  fref[k] = fhi;
            }
         }
      }

      // compare the bias energy and slope over the lambda range
      for (int i = 0; i <= 80; ++i) {
         const double lmda = (double)i * 0.0125;
         CAPTURE(lmda);
         double eb, db, er, dr;
         efreeBias(lmda, eb, db);
         efreeProf(lmda, fref, er, dr);
         COMPARE_REALS(eb, er, 1.0e-10);
         COMPARE_REALS(db, dr, 1.0e-10);
      }
   }

   // check a few bins of the 0.25 reach directly, at the edge of the low end
   // and of each side of the gap
   double eb, db;
   lmdafillr = 0.25;
   efreeBias(0.045, eb, db);
   COMPARE_REALS(db, 0.0, 1.0e-9);
   efreeBias(0.05, eb, db);
   COMPARE_REALS(db, flo, 1.0e-9);
   efreeBias(0.55, eb, db);
   COMPARE_REALS(db, flo, 1.0e-9);
   efreeBias(0.6, eb, db);
   COMPARE_REALS(db, 0.0, 1.0e-9);
   efreeBias(0.65, eb, db);
   COMPARE_REALS(db, fhi, 1.0e-9);

   // the default reach of 0.1 is 20 bins: bins 41 and 81 are the last ones
   // filled on either side of lambda 0.3, so the segment just inside each holds
   // the data value and the segment just outside is zero
   lmdafillr = 0.1;
   efreeBias(0.2025, eb, db);
   REQUIRE(db == flo);
   efreeBias(0.1925, eb, db);
   REQUIRE(db == 0.0);
   efreeBias(0.3975, eb, db);
   REQUIRE(db == flo);
   efreeBias(0.4075, eb, db);
   REQUIRE(db == 0.0);

   // a reach of 0.3 on 21 bins is 0.3/0.05 = 5.999... bins in floating point,
   // which efreeBias must round to 6, so bins 5 and 17 are filled from data at
   // bin 11 and bins 4 and 18 are not
   resetabf(21, 1);
   use_lmdafill = true;
   lmdafillr = 0.3;
   testSetLmdaBin(11, flo);
   efreeBias(0.225, eb, db);
   REQUIRE(db == flo);
   efreeBias(0.125, eb, db);
   REQUIRE(db == 0.0);
   efreeBias(0.775, eb, db);
   REQUIRE(db == flo);
   efreeBias(0.875, eb, db);
   REQUIRE(db == 0.0);
}

// A history saved through the mdsave copies is continued by initabffile and
// abfFromFortran on an interval boundary, its deposits numbered on from the
// steps of the first run (test_eabf.f:test_eabf_restart).
TEST_CASE("EABF-restart", "[ff][eabf][L2]")
{
   const char* argv[] = {"dummy"};
   tinkerFortranRuntimeBegin(1, (char**)argv);
   initial();

   // Ends the runtime and puts back the base name and the ABF flags as the
   // case leaves scope, however it leaves.
   struct Teardown
   {
      char savedname[240];
      int savedleng;
      std::string abffile;
      ~Teardown()
      {
         try {
            if (abffile.size())
               fileExistsAndDelete(abffile);
            std::memcpy(files::filename, savedname, 240);
            files::leng = savedleng;
            dlmda::use_abf = 0;
            dlmda::use_abfdyn = 0;
            testEnd();
         } catch (...) {
         }
      }
   } teardown;
   std::memcpy(teardown.savedname, files::filename, 240);
   teardown.savedleng = files::leng;
   FstrView fname = files::filename;
   fname = "abfsave_tmp";
   files::leng = 11;

   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   lmdaintv = 4;
   lmdanpa = 0;
   lmdanpb = 0;
   lmdanpc = 4;
   lmdadfdl = 0.0;

   // the Fortran state mutate.f would have set up
   bath::kelvin = 300.0;
   dlmda::use_abf = 1;
   dlmda::use_abfdyn = 1;
   dlmda::nlmda = nlmda;
   dlmda::wlmda = wlmda;
   dlmda::wlmda2 = wlmda2;
   dlmda::lmdaintv = lmdaintv;
   dlmda::lmdanpa = lmdanpa;
   dlmda::lmdanpb = lmdanpb;
   dlmda::lmdanpc = lmdanpc;
   dlmda::lmdastep = 0;
   tinker_f_mutate_abf();
   tinker_f_initabffile();
   teardown.abffile = FstrView(dlmda::lmdasavefile).trim();
   abfFromFortran();
   REQUIRE(lmdastep == 0);

   // two intervals and part of a third, then a save
   auto run = [](int nstep, double lmda, double dudl) {
      for (int i = 0; i < nstep; ++i) {
         lambda = lmda;
         dedl = dudl;
         elmdaDyn();
      }
   };
   run(lmdaintv, 0.25, 1.0);
   run(lmdaintv, 0.75, 5.0);
   run(lmdaintv + 2, 0.75, 7.0);
   REQUIRE(nlmdahist == 3);
   REQUIRE(lmdastep == 14);
   for (int k = 1; k <= 3; ++k)
      REQUIRE(lmdaihist[k] == 4 * k);
   mdsaveLmdaFinal(0);
   REQUIRE(dlmda::lmdastep == 14);

   // a fresh run continues the file at the last interval boundary
   dlmda::lmdastep = 0;
   tinker_f_mutate_abf();
   tinker_f_initabffile();
   REQUIRE(FstrView(dlmda::lmdasavefile).trim() == teardown.abffile);
   abfFromFortran();
   REQUIRE(nlmdahist == 3);
   REQUIRE(lmdastep == 12);
   for (int k = 1; k <= 3; ++k)
      REQUIRE(lmdaihist[k] == 4 * k);
   run(lmdaintv, 0.75, 7.0);
   REQUIRE(nlmdahist == 4);
   REQUIRE(lmdaihist[4] == 16);
}
