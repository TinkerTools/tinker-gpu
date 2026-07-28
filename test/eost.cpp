#include "ff/eost.h"
#include "ff/ost.h"
#include "math/const.h"
#include "test.h"
#include "testrt.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/units.hh>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Faithful 1:1 port of the Fortran OST unit-test suite tinker/test/test_eost.f.
// It seeds the OST histogram/kernel state (namespace tinker) with
// deterministic inputs and checks each engine routine against hand-computed
// values. Pure host math -- no molecular system, no GPU; only kelvin is set.
//
// The expected values and tolerances are copied verbatim from test_eost.f. A
// failing REQUIRE indicates a real discrepancy in src/eost.cpp, to be fixed
// there -- do not adjust the expected values.

using namespace tinker;

namespace {
// 2-D accessors so the port reads like the Fortran gkernel(i,j) etc.
double& GK(int i, int j) { return gkernel[gidx(i, j)]; }
double& GF(int i, int j) { return gfkernel[gidx(i, j)]; }
double& GL(int i, int j) { return glkernel[gidx(i, j)]; }
double& GLF(int i, int j) { return glfkernel[gidx(i, j)]; }
int& HEAD(int i, int j) { return osthead[gidx(i, j)]; }

// resetost -- allocate OST arrays and set scalar state to unit-test defaults
// (test_eost.f:1035).
void resetost(int nl, int nf, int nhist)
{
   nlmda = nl;
   nflmda = nf;
   fli0 = (nflmda + 1) / 2;
   wlmda = 1.0 / (double)(nlmda - 1);
   wflmda = 1.0;
   wlmda2 = 0.5 * wlmda;
   wflmda2 = 0.5 * wflmda;
   wlhist = 0.005;
   wfhist = 1.0;
   maxwlhist = wlhist;
   maxwfhist = wfhist;
   nosthist = 0;
   sizeosthist = nhist;
   iosthist = 10;
   ostnequil = 5;
   ostnavg = 5;
   ostlambda = 0.0;
   ostlambdaavg = 0.0;
   ostlambdastd = 0.0;
   ostlambdaslp = 0.0;
   ostdedl = 0.0;
   ostdedlavg = 0.0;
   ostdedlstd = 0.0;
   ostdedlslp = 0.0;
   deffdl = 0.0;
   plmdamap = Lmdamap::QNT;
   elmdamap = Lmdamap::QNT;
   vlmdamap = Lmdamap::QNT;
   plmdaexp = 1;
   elmdaexp = 1;
   vlmdaexp = 1;
   plmdainvn = 1;
   elmdainvn = 1;
   vlmdainvn = 1;
   plmdainveps = 0.0;
   elmdainveps = 0.0;
   vlmdainveps = 0.0;
   osteqratio = 0.5;
   hbias = 0.0;
   eosttot = 0.0;
   oststdev = 1.0;
   ostinterpol = false;
   // tempering off by default, so every pre-existing case keeps the untempered
   // behavior regardless of the (randomized) case order.
   ostemper = false;
   temperthresh = 1.0;
   tempergamma = 1.0;

   osthist.assign(sizeosthist + 1, 0);
   ostihist.assign(sizeosthist + 1, 0);
   ostnext.assign(sizeosthist + 1, 0);
   ostlhist.assign(sizeosthist + 1, 0.0);
   ostfhist.assign(sizeosthist + 1, 0.0);
   osthhist.assign(sizeosthist + 1, 0.0);
   ostwlhist.assign(sizeosthist + 1, 0.0);
   ostwfhist.assign(sizeosthist + 1, 0.0);
   ostllist.assign(iosthist, 0.0);
   ostflist.assign(iosthist, 0.0);
   osthead.assign((size_t)nlmda * nflmda, 0);
   gkernel.assign((size_t)nlmda * nflmda, 0.0);
   gfkernel.assign((size_t)nlmda * nflmda, 0.0);
   glfkernel.assign((size_t)nlmda * nflmda, 0.0);
   glkernel.assign((size_t)nlmda * nflmda, 0.0);
   fkernel.assign(nlmda + 1, 0.0);
   fsumkernel.assign(nlmda + 1, 0.0);
   pfkernel.assign(nlmda + 1, 0.0);
   vkernelmax.assign(nlmda + 1, 0.0);
}

// resetmeta -- allocate metadynamics history arrays (test_eost.f:1170).
void resetmeta(int nhist)
{
   sizemetahist = nhist;
   nmetahist = 0;
   metalhist.assign(sizemetahist + 1, 0.0);
   metahhist.assign(sizemetahist + 1, 0.0);
   metawhist.assign(sizemetahist + 1, 0.0);
   metaihist.assign(sizemetahist + 1, 0);
   // sized off nlmda, so resetost must run first
   vmetagrid.assign(nlmda + 1, 0.0);
   dvmetagrid.assign(nlmda + 1, 0.0);
}

// sethist -- store one gaussian history entry and its packed bin
// (test_eost.f:1324).
void sethist(int ihist, double lambda, double flmda, double height, double sigl, double sigf)
{
   int ilmda = lambdaBin(lambda);
   int iflmda = flambdaBin(flmda);
   int k;
   ijToK(ilmda, iflmda, nlmda, k);
   osthist[ihist] = k;
   ostlhist[ihist] = lambda;
   ostfhist[ihist] = flmda;
   osthhist[ihist] = height;
   ostwlhist[ihist] = sigl;
   ostwfhist[ihist] = sigf;
   maxwlhist = std::max(maxwlhist, sigl);
   maxwfhist = std::max(maxwfhist, sigf);
   ostnext[ihist] = 0;
}

// brutevkmax -- max of the g kernel over the whole flambda axis for one lambda
// bin, computed the slow way. Seeded at 0 to match vkernelmax, whose untouched
// bins stay 0 (deposited heights are positive, so no cell is ever negative).
double brutevkmax(int i)
{
   double m = 0.0;
   for (int j = 1; j <= nflmda; ++j)
      m = std::max(m, GK(i, j));
   return m;
}

// kerneltag -- unique numeric tag for a kernel array and bin (test_eost.f:1237).
double kerneltag(int ikern, int i, int j)
{
   return 1000.0 * (double)ikern + 10.0 * (double)i + (double)j;
}

// seedkernels -- fill each flambda-dependent kernel with index tags
// (test_eost.f:1207).
void seedkernels()
{
   for (int i = 1; i <= nlmda; ++i) {
      for (int j = 1; j <= nflmda; ++j) {
         GK(i, j) = kerneltag(1, i, j);
         GF(i, j) = kerneltag(2, i, j);
         GL(i, j) = kerneltag(3, i, j);
         GLF(i, j) = kerneltag(4, i, j);
      }
   }
}

// checkkernels -- compare resized kernels against references whose seeded block
// has shifted by offset bins (test_eost.f:1259).
void checkkernels(const std::string& label, int nold, int offset)
{
   INFO(label);
   for (int i = 1; i <= nlmda; ++i) {
      for (int j = 1; j <= nold; ++j) {
         CAPTURE(i, j);
         COMPARE_REALS(GK(i, j + offset), kerneltag(1, i, j), 1.0e-12);
         COMPARE_REALS(GF(i, j + offset), kerneltag(2, i, j), 1.0e-12);
         COMPARE_REALS(GL(i, j + offset), kerneltag(3, i, j), 1.0e-12);
         COMPARE_REALS(GLF(i, j + offset), kerneltag(4, i, j), 1.0e-12);
      }
   }
   // bins outside the copied block must be zero after the resize
   for (int i = 1; i <= nlmda; ++i) {
      for (int j = 1; j <= nflmda; ++j) {
         bool inblock = (j > offset && j - offset <= nold);
         if (!inblock) {
            CAPTURE(i, j);
            COMPARE_REALS(GK(i, j), 0.0, 1.0e-12);
            COMPARE_REALS(GF(i, j), 0.0, 1.0e-12);
            COMPARE_REALS(GL(i, j), 0.0, 1.0e-12);
            COMPARE_REALS(GLF(i, j), 0.0, 1.0e-12);
         }
      }
   }
}
}

TEST_CASE("EOST-index", "[ff][eost]")
{
   int i, j, k;
   ijToK(3, 4, 7, k);
   COMPARE_INTS(k, 24);
   kToIj(24, 7, i, j);
   COMPARE_INTS(i, 3);
   COMPARE_INTS(j, 4);
}

TEST_CASE("EOST-resize", "[ff][eost]")
{
   resetost(5, 5, 2);
   nosthist = 2;
   for (int i = 1; i <= 2; ++i) {
      osthist[i] = 10 + i;
      ostihist[i] = 100 + i;
      ostnext[i] = i - 1;
      ostlhist[i] = 0.25 * (double)i;
      ostfhist[i] = -3.0 + 2.0 * (double)i;
      osthhist[i] = 1.0 + (double)i;
      ostwlhist[i] = 0.25;
      ostwfhist[i] = 1.0;
   }
   resizeOstHist();
   COMPARE_INTS(sizeosthist, 4);
   for (int i = 1; i <= 4; ++i) {
      CAPTURE(i);
      if (i <= 2) {
         COMPARE_INTS(osthist[i], 10 + i);
         COMPARE_INTS(ostihist[i], 100 + i);
         COMPARE_INTS(ostnext[i], i - 1);
         COMPARE_REALS(ostlhist[i], 0.25 * (double)i, 1.0e-12);
         COMPARE_REALS(ostfhist[i], -3.0 + 2.0 * (double)i, 1.0e-12);
         COMPARE_REALS(osthhist[i], 1.0 + (double)i, 1.0e-12);
         COMPARE_REALS(ostwlhist[i], 0.25, 1.0e-12);
         COMPARE_REALS(ostwfhist[i], 1.0, 1.0e-12);
      } else {
         COMPARE_INTS(osthist[i], 0);
         COMPARE_INTS(ostihist[i], 0);
         COMPARE_INTS(ostnext[i], 0);
         COMPARE_REALS(ostlhist[i], 0.0, 1.0e-12);
         COMPARE_REALS(ostfhist[i], 0.0, 1.0e-12);
         COMPARE_REALS(osthhist[i], 0.0, 1.0e-12);
         COMPARE_REALS(ostwlhist[i], 0.0, 1.0e-12);
         COMPARE_REALS(ostwfhist[i], 0.0, 1.0e-12);
      }
   }
}

TEST_CASE("EOST-buildindex", "[ff][eost]")
{
   resetost(5, 5, 3);
   nosthist = 3;
   sethist(1, 0.50, 0.0, 1.0, wlmda, wflmda);
   sethist(2, 0.50, 0.0, 2.0, wlmda, wflmda);
   sethist(3, 0.75, 1.0, 3.0, wlmda, wflmda);
   buildOstIndex();
   int ilmda = 3, iflmda = 3, k;
   ijToK(ilmda, iflmda, nlmda, k);
   COMPARE_INTS(osthist[1], k);
   COMPARE_INTS(HEAD(ilmda, iflmda), 2);
   COMPARE_INTS(ostnext[2], 1);
   COMPARE_INTS(ostnext[1], 0);
}

TEST_CASE("EOST-ensure", "[ff][eost]")
{
   int k, nold, oldfli0, ilmda, iflmda;

   // high-side expansion preserves fli0 and old gkernel values
   resetost(3, 5, 1);
   GK(2, 3) = 7.0;
   ensureFlambda(2000.0);
   COMPARE_INTS(nflmda, 2105);
   COMPARE_INTS(fli0, 3);
   COMPARE_REALS(GK(2, 3), 7.0, 1.0e-12);

   // low-side expansion shifts fli0 and old gkernel values
   resetost(3, 5, 1);
   GK(2, 3) = 7.0;
   ensureFlambda(-2000.0);
   COMPARE_INTS(nflmda, 2105);
   COMPARE_INTS(fli0, 2103);
   COMPARE_REALS(GK(2, 2103), 7.0, 1.0e-12);

   // low-side expansion also rebuilds osthead, ostnext and osthist
   resetost(3, 5, 2);
   nosthist = 2;
   sethist(1, 0.50, 0.0, 1.0, wlmda, wflmda);
   sethist(2, 0.50, 0.0, 2.0, wlmda, wflmda);
   buildOstIndex();
   ensureFlambda(-2000.0);
   ilmda = 2;
   iflmda = 2103;
   ijToK(ilmda, iflmda, nlmda, k);
   COMPARE_INTS(osthist[1], k);
   COMPARE_INTS(osthist[2], k);
   COMPARE_INTS(HEAD(ilmda, iflmda), 2);
   COMPARE_INTS(ostnext[2], 1);
   COMPARE_INTS(ostnext[1], 0);

   // the resize copies four separate kernels; seed all, then require every
   // entry to land where it belongs
   resetost(3, 5, 1);
   nold = nflmda;
   oldfli0 = fli0;
   seedkernels();
   ensureFlambda(2000.0);
   checkkernels("ensureflambda high", nold, fli0 - oldfli0);
   resetost(3, 5, 1);
   nold = nflmda;
   oldfli0 = fli0;
   seedkernels();
   ensureFlambda(-2000.0);
   checkkernels("ensureflambda low", nold, fli0 - oldfli0);
}

TEST_CASE("EOST-gkernels", "[ff][eost]")
{
   double egbias, dgdl, dgdfl, expected, height;

   // choose height so the normalized gaussian prefactor is one
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 0.50, 0.0, height, wlmda, wflmda);
   buildOstIndex();

   // addgkernelhist from a single interior gaussian
   addGkernelHist(1);
   COMPARE_REALS(GK(3, 3), 1.0, 1.0e-12);
   expected = std::exp(-1.0);
   COMPARE_REALS(GK(4, 4), expected, 1.0e-12);

   // addgkernelhist includes left-boundary mirror image
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 0.0, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   addGkernelHist(1);
   COMPARE_REALS(GK(1, 3), 2.0, 1.0e-12);
   expected = 2.0 * std::exp(-0.5);
   COMPARE_REALS(GK(2, 3), expected, 1.0e-12);

   // addgkernelhist includes right-boundary mirror image
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 1.0, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   addGkernelHist(1);
   COMPARE_REALS(GK(5, 3), 2.0, 1.0e-12);
   expected = 2.0 * std::exp(-0.5);
   COMPARE_REALS(GK(4, 3), expected, 1.0e-12);

   // rebuild the original interior gaussian for later checks
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 0.50, 0.0, height, wlmda, wflmda);
   buildOstIndex();

   // buildgkernel rebuilds the same full grid
   buildGkernel();
   COMPARE_REALS(GK(3, 3), 1.0, 1.0e-12);
   expected = std::exp(-1.0);
   COMPARE_REALS(GK(4, 4), expected, 1.0e-12);

   // updategkernel adds only the newest gaussian to current grid
   std::fill(gkernel.begin(), gkernel.end(), 0.0);
   updateGkernel();
   COMPARE_REALS(GK(3, 3), 1.0, 1.0e-12);

   // adding a taller gaussian updates only from the new history entry
   nosthist = 2;
   sethist(2, 0.75, 1.0, 2.0 * height, wlmda, wflmda);
   buildOstIndex();
   updateGkernel();
   expected = 2.0 + std::exp(-1.0);
   COMPARE_REALS(GK(4, 4), expected, 1.0e-12);
   expected = 1.0 + 2.0 * std::exp(-1.0);
   COMPARE_REALS(GK(3, 3), expected, 1.0e-12);

   // egkernel evaluates the continuous gaussian and derivatives
   ostlambda = 0.75;
   ostdedl = 1.0;
   egkernel(egbias, dgdl, dgdfl);
   expected = 2.0 + std::exp(-1.0);
   COMPARE_REALS(egbias, expected, 1.0e-12);
   COMPARE_REALS(dgdl, -4.0 * std::exp(-1.0), 1.0e-12);
   COMPARE_REALS(dgdfl, -std::exp(-1.0), 1.0e-12);

   // two gaussians in the same bin are both followed by ostnext
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 2;
   sethist(1, 0.50, 0.0, height, wlmda, wflmda);
   sethist(2, 0.50, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   ostlambda = 0.50;
   ostdedl = 0.0;
   egkernel(egbias, dgdl, dgdfl);
   COMPARE_REALS(egbias, 2.0, 1.0e-12);
   COMPARE_REALS(dgdl, 0.0, 1.0e-12);
   COMPARE_REALS(dgdfl, 0.0, 1.0e-12);

   // multiple bins are found through osthead lookup
   resetost(5, 5, 4);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 3;
   sethist(1, 0.50, 0.0, 1.0 * height, wlmda, wflmda);
   sethist(2, 0.50, 0.0, 2.0 * height, wlmda, wflmda);
   sethist(3, 0.75, 1.0, 3.0 * height, wlmda, wflmda);
   buildOstIndex();
   ostlambda = 0.75;
   ostdedl = 1.0;
   egkernel(egbias, dgdl, dgdfl);
   expected = 3.0 + 3.0 * std::exp(-1.0);
   COMPARE_REALS(egbias, expected, 1.0e-12);
   COMPARE_REALS(dgdl, -12.0 * std::exp(-1.0), 1.0e-12);
   COMPARE_REALS(dgdfl, -3.0 * std::exp(-1.0), 1.0e-12);

   // left endpoint includes both real and mirror gaussian images
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 0.0, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   ostlambda = 0.0;
   ostdedl = 0.0;
   egkernel(egbias, dgdl, dgdfl);
   COMPARE_REALS(egbias, 2.0, 1.0e-12);
   COMPARE_REALS(dgdl, 0.0, 1.0e-12);

   // right endpoint includes both real and mirror gaussian images
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 1.0, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   ostlambda = 1.0;
   ostdedl = 0.0;
   egkernel(egbias, dgdl, dgdfl);
   COMPARE_REALS(egbias, 2.0, 1.0e-12);
   COMPARE_REALS(dgdl, 0.0, 1.0e-12);

   // wide histogram widths are used for both grid and continuous bias
   resetost(41, 81, 3);
   wlhist = 0.05;
   wfhist = 10.0;
   maxwlhist = wlhist;
   maxwfhist = wfhist;
   height = 2.0 * pi * wlhist * wfhist;
   {
      double targetl = 0.525;
      double targetf = 5.0;
      nosthist = 1;
      sethist(1, 0.50, 0.0, height, wlhist, wfhist);
      buildOstIndex();
      buildGkernel();
      expected = std::exp(-0.25);
      COMPARE_REALS(GK(22, 46), expected, 1.0e-12);
      ostlambda = targetl;
      ostdedl = targetf;
      egkernel(egbias, dgdl, dgdfl);
      COMPARE_REALS(egbias, expected, 1.0e-12);
      COMPARE_REALS(dgdl, -10.0 * expected, 1.0e-12);
      COMPARE_REALS(dgdfl, -0.05 * expected, 1.0e-12);
   }

   // if dU/dlambda is outside the current flambda grid, egkernel returns zero
   resetost(5, 5, 3);
   height = 2.0 * pi * wlmda * wflmda;
   nosthist = 1;
   sethist(1, 0.50, 0.0, height, wlmda, wflmda);
   buildOstIndex();
   ostlambda = 0.50;
   ostdedl = 100.0;
   egkernel(egbias, dgdl, dgdfl);
   COMPARE_REALS(egbias, 0.0, 1.0e-12);
   COMPARE_REALS(dgdl, 0.0, 1.0e-12);
   COMPARE_REALS(dgdfl, 0.0, 1.0e-12);
}

TEST_CASE("EOST-fkernel", "[ff][eost]")
{
   bath::kelvin = 300.0;
   double rt, expected, height;

   // two nonzero gkernel weights at flambda=-1 and +1
   resetost(5, 5, 1);
   rt = units::gasconst * bath::kelvin;
   GK(3, 2) = std::log(2.0) * rt;
   GK(3, 4) = std::log(4.0) * rt;
   buildFkernel();
   expected = 1.0 / 3.0;
   COMPARE_REALS(fkernel[3], expected, 1.0e-12);
   COMPARE_REALS(fkernel[1], 0.0, 1.0e-12);

   // build gkernel incrementally from multiple gaussians, then build f kernel
   resetost(5, 5, 4);
   rt = units::gasconst * bath::kelvin;
   height = 2.0 * pi * wlmda * wflmda;

   nosthist = 1;
   sethist(1, 0.25, -1.0, 1.0 * height, wlmda, wflmda);
   buildOstIndex();
   updateGkernel();

   nosthist = 2;
   sethist(2, 0.50, 0.0, 2.0 * height, wlmda, wflmda);
   buildOstIndex();
   updateGkernel();

   nosthist = 3;
   sethist(3, 0.75, 1.0, 3.0 * height, wlmda, wflmda);
   buildOstIndex();
   updateGkernel();

   buildFkernel();

   double w1 = std::exp(GK(3, 1) / rt);
   double w2 = std::exp(GK(3, 2) / rt);
   double w3 = std::exp(GK(3, 3) / rt);
   double w4 = std::exp(GK(3, 4) / rt);
   double w5 = std::exp(GK(3, 5) / rt);
   expected = (-2.0 * w1 - w2 + w4 + 2.0 * w5) / (w1 + w2 + w3 + w4 + w5);
   COMPARE_REALS(fkernel[3], expected, 1.0e-12);
}

TEST_CASE("EOST-kernelbuilds", "[ff][eost]")
{
   bath::kelvin = 300.0;
   double rt, height;
   int nhist;

   // build a mixed history with overlapping gaussians, endpoint mirror images,
   // and multiple gaussian widths
   resetost(9, 9, 8);
   rt = units::gasconst * bath::kelvin;
   nhist = 6;
   nosthist = nhist;
   height = 2.0 * pi * wlmda * wflmda;
   sethist(1, 0.00, 0.0, 0.7 * height, wlmda, wflmda);
   sethist(2, 0.25, -1.0, 1.1 * height, wlmda, wflmda);
   sethist(3, 0.50, 0.0, 1.6 * height, wlmda, wflmda);
   sethist(4, 0.50, 0.0, 0.4 * height, wlmda, wflmda);
   sethist(5, 0.75, 1.0, 2.3 * height, wlmda, wflmda);
   sethist(6, 1.00, 0.0, 0.9 * height, wlmda, wflmda);
   ostwlhist[5] = 2.0 * wlmda;
   ostwfhist[5] = 2.0 * wflmda;
   buildOstIndex();

   // buildgkernel must clear stale values before rebuilding
   std::fill(gkernel.begin(), gkernel.end(), -123.0);
   buildGkernel();
   buildFkernel();

   std::vector<double> gref = gkernel;
   std::vector<double> fref(nlmda + 1, 0.0);
   for (int i = 1; i <= nlmda; ++i)
      fref[i] = fkernel[i];
   COMPARE_REALS(GK(1, 1), 0.0, 1.0e-12);

   // compute the buildfkernel result independently from gkernel
   std::vector<double> fmanual(nlmda + 1, 0.0);
   std::vector<double> fsumref(nlmda + 1, 0.0);
   std::vector<double> pfref(nlmda + 1, 0.0);
   for (int i = 1; i <= nlmda; ++i) {
      double partfunc = 0.0, fsum = 0.0;
      for (int j = 1; j <= nflmda; ++j) {
         if (gref[gidx(i, j)] != 0.0) {
            double flmda = (double)(j - fli0) * wflmda;
            double weight = std::exp(gref[gidx(i, j)] / rt);
            fsum += flmda * weight;
            partfunc += weight;
         }
      }
      fmanual[i] = (partfunc == 0.0) ? 0.0 : fsum / partfunc;
      fsumref[i] = fsum;
      pfref[i] = partfunc;
   }
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(fkernel[i], fmanual[i], 1.0e-12);
   }

   // buildkernels must reproduce gkernel, fkernel and the accumulators
   std::fill(gkernel.begin(), gkernel.end(), -123.0);
   for (int i = 1; i <= nlmda; ++i) {
      fkernel[i] = -123.0;
      fsumkernel[i] = -123.0;
      pfkernel[i] = -123.0;
   }
   buildKernels();
   for (int j = 1; j <= nflmda; ++j) {
      for (int i = 1; i <= nlmda; ++i) {
         CAPTURE(i, j);
         COMPARE_REALS(GK(i, j), gref[gidx(i, j)], 1.0e-12);
      }
   }
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(fkernel[i], fref[i], 1.0e-12);
      COMPARE_REALS(fsumkernel[i], fsumref[i], 1.0e-12);
      COMPARE_REALS(pfkernel[i], pfref[i], 1.0e-12);
   }

   // incremental updates one history at a time must match a full rebuild
   resetost(9, 9, 8);
   height = 2.0 * pi * wlmda * wflmda;
   for (int ihist = 1; ihist <= nhist; ++ihist) {
      nosthist = ihist;
      if (ihist == 1)
         sethist(ihist, 0.00, 0.0, 0.7 * height, wlmda, wflmda);
      else if (ihist == 2)
         sethist(ihist, 0.25, -1.0, 1.1 * height, wlmda, wflmda);
      else if (ihist == 3)
         sethist(ihist, 0.50, 0.0, 1.6 * height, wlmda, wflmda);
      else if (ihist == 4)
         sethist(ihist, 0.50, 0.0, 0.4 * height, wlmda, wflmda);
      else if (ihist == 5) {
         sethist(ihist, 0.75, 1.0, 2.3 * height, wlmda, wflmda);
         ostwlhist[ihist] = 2.0 * wlmda;
         ostwfhist[ihist] = 2.0 * wflmda;
      } else
         sethist(ihist, 1.00, 0.0, 0.9 * height, wlmda, wflmda);
      buildOstIndex();
      updateKernels();
   }
   for (int j = 1; j <= nflmda; ++j) {
      for (int i = 1; i <= nlmda; ++i) {
         CAPTURE(i, j);
         COMPARE_REALS(GK(i, j), gref[gidx(i, j)], 1.0e-12);
      }
   }
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(fkernel[i], fref[i], 1.0e-12);
      COMPARE_REALS(fsumkernel[i], fsumref[i], 1.0e-12);
      COMPARE_REALS(pfkernel[i], pfref[i], 1.0e-12);
   }
}

TEST_CASE("EOST-histstat", "[ff][eost]")
{
   resetost(5, 5, 1);
   iosthist = 6;
   ostnequil = 2;
   ostnavg = 4;
   // samples 1..6 laid out 0-based, so the slice still holds the values 3..6
   for (int i = 0; i < iosthist; ++i) {
      ostllist[i] = (double)(i + 1);
      ostflist[i] = 2.0 * (double)(i + 1);
   }

   // the whole-slice statistics cover list[2..5], the values 3..6
   ostcvbin = 2;
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   histstat(ostflist, ostdedlavg, ostdedlstd, ostdedlslp, ostdedlavgbin, ostdedlstdbin, ostdedlslpbin);
   double stdref = std::sqrt(1.25);
   COMPARE_REALS(ostlambdaavg, 4.5, 1.0e-12);
   COMPARE_REALS(ostdedlavg, 9.0, 1.0e-12);
   COMPARE_REALS(ostlambdastd, stdref, 1.0e-12);
   COMPARE_REALS(ostdedlstd, 2.0 * stdref, 1.0e-12);

   // fitted changes per sample preserve the scale of each ramp
   COMPARE_REALS(ostlambdaslp, 1.0, 1.0e-12);
   COMPARE_REALS(ostdedlslp, 2.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[0], 1.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[1], 1.0, 1.0e-12);
   COMPARE_REALS(ostdedlslpbin[0], 2.0, 1.0e-12);
   COMPARE_REALS(ostdedlslpbin[1], 2.0, 1.0e-12);

   // 4 samples into 2 bins divides evenly: values {3,4} and {5,6}
   REQUIRE((int)ostlambdaavgbin.size() == 2);
   COMPARE_REALS(ostlambdaavgbin[0], 3.5, 1.0e-12);
   COMPARE_REALS(ostlambdaavgbin[1], 5.5, 1.0e-12);
   COMPARE_REALS(ostlambdastdbin[0], 0.5, 1.0e-12);
   COMPARE_REALS(ostlambdastdbin[1], 0.5, 1.0e-12);
   COMPARE_REALS(ostdedlavgbin[0], 7.0, 1.0e-12);
   COMPARE_REALS(ostdedlavgbin[1], 11.0, 1.0e-12);

   // 4 samples into 3 bins keeps 1 per bin and drops the leading value 3
   ostcvbin = 3;
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   REQUIRE((int)ostlambdaavgbin.size() == 3);
   COMPARE_REALS(ostlambdaavgbin[0], 4.0, 1.0e-12);
   COMPARE_REALS(ostlambdaavgbin[1], 5.0, 1.0e-12);
   COMPARE_REALS(ostlambdaavgbin[2], 6.0, 1.0e-12);
   for (int b = 0; b < 3; ++b) {
      COMPARE_REALS(ostlambdastdbin[b], 0.0, 1.0e-12);
      COMPARE_REALS(ostlambdaslpbin[b], 0.0, 1.0e-12); // single-sample bins
   }
   COMPARE_REALS(ostlambdaslp, 1.0, 1.0e-12); // the whole slice still ramps
}

TEST_CASE("EOST-histstat-drift", "[ff][eost]")
{
   resetost(5, 5, 1);
   iosthist = 8;
   ostnequil = 0;
   ostnavg = 8;
   ostcvbin = 2;

   // a flat series has zero slope
   for (int i = 0; i < iosthist; ++i)
      ostllist[i] = 7.0;
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   COMPARE_REALS(ostlambdaavg, 7.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslp, 0.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[0], 0.0, 1.0e-12);

   // a strictly decreasing ramp retains its fitted change per sample
   for (int i = 0; i < iosthist; ++i)
      ostllist[i] = -0.5 * (double)i;
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   COMPARE_REALS(ostlambdaslp, -0.5, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[0], -0.5, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[1], -0.5, 1.0e-12);

   // a symmetric V has no net drift overall, but each half drifts fully
   double v[8] = {4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0};
   for (int i = 0; i < iosthist; ++i)
      ostllist[i] = v[i];
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   COMPARE_REALS(ostlambdaslp, 0.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[0], -1.0, 1.0e-12);
   COMPARE_REALS(ostlambdaslpbin[1], 1.0, 1.0e-12);

   // accuracy: a large offset must not swamp a small drift (the K shift)
   for (int i = 0; i < iosthist; ++i)
      ostllist[i] = 5000.0 + 1.0e-6 * (double)i;
   histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
      ostlambdaslpbin);
   COMPARE_REALS(ostlambdaslp, 1.0e-6, 1.0e-9);
}

TEST_CASE("EOST-depcriteria", "[ff][eost]")
{
   ostcvstd = 10.0;
   ostcvslp = 0.5;
   ostcvdif = 4.0;
   ostcvrat = 0.2;

   std::vector<double> avgbin = {2.0, 4.0, 6.0};
   REQUIRE(depcriteria(50.0, 10.0, 0.5, avgbin));

   REQUIRE_FALSE(depcriteria(100.0, 10.1, 0.0, avgbin));
   REQUIRE_FALSE(depcriteria(10.0, 2.1, 0.0, avgbin));
   REQUIRE_FALSE(depcriteria(-10.0, 2.1, 0.0, avgbin));
   REQUIRE_FALSE(depcriteria(0.0, 1.0, 0.0, avgbin));
   REQUIRE(depcriteria(0.0, 0.0, 0.0, avgbin));
   REQUIRE_FALSE(depcriteria(100.0, 0.0, 0.6, avgbin));
   REQUIRE_FALSE(depcriteria(100.0, 0.0, -0.6, avgbin));

   avgbin.back() = 6.1;
   REQUIRE_FALSE(depcriteria(100.0, 0.0, 0.0, avgbin));
}

TEST_CASE("EOST-depcriteria2", "[ff][eost]")
{
   ostcvstd = 10.0;
   ostcvrat = 0.2;

   REQUIRE(depcriteria2(0.0, 9.9));
   REQUIRE_FALSE(depcriteria2(0.0, 10.0));
   REQUIRE(depcriteria2(50.0, 19.9));
   REQUIRE_FALSE(depcriteria2(50.0, 20.0));
   REQUIRE(depcriteria2(-50.0, 19.9));
   REQUIRE_FALSE(depcriteria2(-50.0, 20.0));
}

TEST_CASE("EOST-eginterpolate", "[ff][eost]")
{
   bath::kelvin = 300.0;
   double egbias0, dgdl0, dgdfl0;
   double egbias1, dgdl1, dgdfl1;
   double height, sigl, sigf;

   // grid-point interpolation reproduces the analytic gaussian sum to roundoff
   resetost(9, 9, 4);
   sigl = 2.0 * wlmda;
   sigf = 2.0 * wflmda;
   height = 2.0 * pi * sigl * sigf;
   oststdev = 4.0;
   nosthist = 3;
   sethist(1, 0.25, -1.0, 1.1 * height, sigl, sigf);
   sethist(2, 0.50, 0.0, 1.6 * height, sigl, sigf);
   sethist(3, 0.75, 1.0, 2.3 * height, sigl, sigf);
   buildOstIndex();
   buildKernels();
   ostlambda = 0.50;
   ostdedl = 0.0;
   ostinterpol = false;
   egkernel(egbias0, dgdl0, dgdfl0);
   egkernelInterpolate(egbias1, dgdl1, dgdfl1);
   COMPARE_REALS(egbias1, egbias0, 1.0e-12);
   COMPARE_REALS(dgdl1, dgdl0, 1.0e-12);
   COMPARE_REALS(dgdfl1, dgdfl0, 1.0e-12);

   // off-grid interpolation is approximate; wide gaussians keep errors small
   resetost(17, 17, 4);
   sigl = 4.0 * wlmda;
   sigf = 4.0 * wflmda;
   height = 2.0 * pi * sigl * sigf;
   oststdev = 4.0;
   nosthist = 3;
   sethist(1, 0.25, -2.0, 0.8 * height, sigl, sigf);
   sethist(2, 0.50, 0.0, 1.2 * height, sigl, sigf);
   sethist(3, 0.75, 2.0, 1.6 * height, sigl, sigf);
   buildOstIndex();
   buildKernels();
   ostlambda = 0.53125;
   ostdedl = 0.25;
   ostinterpol = false;
   egkernel(egbias0, dgdl0, dgdfl0);
   egkernelInterpolate(egbias1, dgdl1, dgdfl1);
   COMPARE_REALS(egbias1, egbias0, 1.0e-3);
   COMPARE_REALS(dgdl1, dgdl0, 2.0e-2);
   COMPARE_REALS(dgdfl1, dgdfl0, 2.0e-2);
}

TEST_CASE("EOST-efkernel", "[ff][eost]")
{
   double eostlmda, dfdl, expected;

   // use fkernel(lambda)=lambda and integrate to lambda=0.375
   resetost(5, 5, 1);
   for (int i = 1; i <= nlmda; ++i)
      fkernel[i] = (double)(i - 1) * wlmda;
   ostlambda = 0.375;
   efkernel(eostlmda, dfdl);
   expected = 0.5 * ostlambda * ostlambda;
   COMPARE_REALS(eostlmda, expected, 1.0e-12);
   COMPARE_REALS(dfdl, ostlambda, 1.0e-12);

   // use fkernel(lambda)=1+lambda so endpoint mean forces are nonzero
   resetost(5, 5, 1);
   for (int i = 1; i <= nlmda; ++i)
      fkernel[i] = 1.0 + (double)(i - 1) * wlmda;

   ostlambda = 0.0;
   efkernel(eostlmda, dfdl);
   COMPARE_REALS(eostlmda, 0.0, 1.0e-12);
   COMPARE_REALS(dfdl, 1.0, 1.0e-12);
   ostlambda = -0.25;
   efkernel(eostlmda, dfdl);
   COMPARE_REALS(eostlmda, 0.0, 1.0e-12);
   COMPARE_REALS(dfdl, 1.0, 1.0e-12);

   ostlambda = 1.0;
   efkernel(eostlmda, dfdl);
   COMPARE_REALS(eostlmda, 1.5, 1.0e-12);
   COMPARE_REALS(dfdl, 2.0, 1.0e-12);
   ostlambda = 1.25;
   efkernel(eostlmda, dfdl);
   COMPARE_REALS(eostlmda, 1.5, 1.0e-12);
   COMPARE_REALS(dfdl, 2.0, 1.0e-12);

   COMPARE_REALS(etotFkernel(), 1.5, 1.0e-12);

   for (int i = 1; i <= nlmda; ++i)
      fkernel[i] = 2.5;
   COMPARE_REALS(etotFkernel(), 2.5, 1.0e-12);

   for (int i = 1; i <= nlmda; ++i)
      fkernel[i] = 0.0;
   COMPARE_REALS(etotFkernel(), 0.0, 1.0e-12);
}

TEST_CASE("EOST-meta", "[ff][eost]")
{
   double vbias, dvdl, pref, expected;

   // one normalized 1D gaussian centered at lambda=0.5
   resetost(5, 5, 1);
   resetmeta(2);
   nmetahist = 1;
   metalhist[1] = 0.5;
   metahhist[1] = 2.0;
   metawhist[1] = 0.25;
   pref = metahhist[1] / (metawhist[1] * std::sqrt(2.0 * pi));

   // the three reflecting images sit at lambda = 0.5, -0.5 and 1.5; sig2 = 0.0625.
   // at the center the two images are equidistant, so they cancel in dvdl.
   eMetaBias(0.5, vbias, dvdl);
   expected = pref * (1.0 + 2.0 * std::exp(-8.0));
   COMPARE_REALS(vbias, expected, 1.0e-12);
   COMPARE_REALS(dvdl, 0.0, 1.0e-12);

   // off center the image offsets are 0.25, 1.25 and -0.75
   eMetaBias(0.75, vbias, dvdl);
   double b1 = pref * std::exp(-0.5);
   double b2 = pref * std::exp(-12.5);
   double b3 = pref * std::exp(-4.5);
   COMPARE_REALS(vbias, b1 + b2 + b3, 1.0e-12);
   COMPARE_REALS(dvdl, -(0.25 * b1 + 1.25 * b2 - 0.75 * b3) / 0.0625, 1.0e-12);

   // symmetric gaussian has zero endpoint free energy difference
   COMPARE_REALS(metaDeltaG(), 0.0, 1.0e-12);

   // resizing preserves old history and zeros new slots
   nmetahist = 2;
   metalhist[2] = 0.25;
   metahhist[2] = 3.0;
   metawhist[2] = 0.125;
   metaihist[2] = 42;
   resizeMeta();
   COMPARE_INTS(sizemetahist, 4);
   COMPARE_REALS(metalhist[2], 0.25, 1.0e-12);
   COMPARE_REALS(metalhist[3], 0.0, 1.0e-12);
   COMPARE_INTS(metaihist[2], 42);
   COMPARE_INTS(metaihist[3], 0);
}

TEST_CASE("EOST-metadyn", "[ff][eost]")
{
   // drive eMetaDyn across one full interval and check the deposited gaussian.
   resetost(5, 5, 1);
   resetmeta(2);
   iosthist = 4;
   ostnequil = 2;
   ostnavg = 2;
   hbias = 2.0;
   wlmda = 0.25;
   ostdedl = 0.0;
   ostdt = 0.0; // no-op ostLangevin, so the sampled lambda values stay controlled

   // sampled lambda per step (lam is indexed by istep, not by buffer slot);
   // histstat averages the post-equilibration slice, indices
   // ostnequil..iosthist-1 = 2..3, holding the last two samples.
   double lam[5] = {0.0, 0.1, 0.2, 0.4, 0.6};
   for (int istep = 1; istep <= iosthist; ++istep) {
      ostlambda = lam[istep];
      eMetaDyn(istep);
      if (istep < iosthist)
         COMPARE_INTS(nmetahist, 0); // no deposit before the interval boundary
   }

   double avgref = (lam[3] + lam[4]) / (double)ostnavg; // 0.5
   COMPARE_INTS(nmetahist, 1);
   COMPARE_REALS(metalhist[1], avgref, 1.0e-12);
   COMPARE_REALS(metahhist[1], hbias, 1.0e-12);
   COMPARE_REALS(metawhist[1], wlmda, 1.0e-12);
   COMPARE_INTS(metaihist[1], iosthist); // step stamp at the deposit boundary
}

TEST_CASE("EOST-vkernelmax", "[ff][eost]")
{
   // vkernelmax must hold max_j gkernel(i,j) for every lambda bin, however the
   // kernel was filled, and ostVminimax must be the minimum of it over lambda.
   bath::kelvin = 300.0;
   resetost(5, 5, 4);

   nosthist = 2;
   sethist(1, 0.25, 0.0, 1.0, 0.25, 1.0);
   sethist(2, 0.75, 1.0, 2.0, 0.25, 1.0);
   buildOstIndex();
   buildKernels();

   double vmin = 1.0e30;
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(vkernelmax[i], brutevkmax(i), 1.0e-12);
      vmin = std::min(vmin, brutevkmax(i));
   }
   COMPARE_REALS(ostVminimax(), vmin, 1.0e-12);
   REQUIRE(vmin > 0.0); // the two sources reach every lambda bin

   // the incremental update through updateKernels must stay exact
   nosthist = 3;
   sethist(3, 0.5, -1.0, 1.5, 0.25, 1.0);
   buildOstIndex();
   updateKernels();
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(vkernelmax[i], brutevkmax(i), 1.0e-12);
   }

   // buildGkernel takes the other gkernel write path (do_f false); it must
   // reproduce both the kernel and the running max
   std::vector<double> gsave = gkernel;
   buildGkernel();
   for (size_t k = 0; k < gsave.size(); ++k) {
      CAPTURE(k);
      COMPARE_REALS(gkernel[k], gsave[k], 1.0e-12);
   }
   for (int i = 1; i <= nlmda; ++i) {
      CAPTURE(i);
      COMPARE_REALS(vkernelmax[i], brutevkmax(i), 1.0e-12);
   }
}

TEST_CASE("EOST-tempering", "[ff][eost]")
{
   bath::kelvin = 300.0;
   resetost(5, 5, 1);
   hbias = 1.0e-5;
   double rt = units::gasconst * bath::kelvin;

   // disabled: the height is hbias no matter how filled the path is
   ostemper = false;
   temperthresh = 1.0;
   tempergamma = 1.0;
   COMPARE_REALS(temperedHeight(0.0), hbias, 1.0e-18);
   COMPARE_REALS(temperedHeight(5.0), hbias, 1.0e-18);
   COMPARE_REALS(temperedHeight(50.0), hbias, 1.0e-18);

   // enabled, at or below the threshold: still exactly hbias
   ostemper = true;
   COMPARE_REALS(temperedHeight(0.0), hbias, 1.0e-18);
   COMPARE_REALS(temperedHeight(0.5), hbias, 1.0e-18);
   COMPARE_REALS(temperedHeight(1.0), hbias, 1.0e-18);

   // above the threshold: exponential decay, strictly monotonic in V*
   double vstar[4] = {1.5, 2.0, 3.0, 5.0};
   double prev = hbias;
   for (int k = 0; k < 4; ++k) {
      CAPTURE(vstar[k]);
      double h = temperedHeight(vstar[k]);
      COMPARE_REALS(h, hbias * std::exp(-(vstar[k] - 1.0) / rt), 1.0e-18);
      REQUIRE(h < prev);
      prev = h;
   }

   // a larger gamma decays more slowly at the same V*
   tempergamma = 1.0;
   double h1 = temperedHeight(3.0);
   tempergamma = 2.0;
   double h2 = temperedHeight(3.0);
   REQUIRE(h2 > h1);
   COMPARE_REALS(h2, hbias * std::exp(-2.0 / (2.0 * rt)), 1.0e-18);

   // a non-positive gamma disables the decay rather than dividing by zero
   tempergamma = 0.0;
   COMPARE_REALS(temperedHeight(50.0), hbias, 1.0e-18);
   tempergamma = -1.0;
   COMPARE_REALS(temperedHeight(50.0), hbias, 1.0e-18);
}

TEST_CASE("EOST-metaimage", "[ff][eost]")
{
   // eMetaBias sums the same three reflecting lambda images as the g kernel.
   double vbias, dvdl;
   resetost(5, 5, 1);
   resetmeta(2);
   nmetahist = 1;
   metalhist[1] = 0.1;
   metahhist[1] = 2.0;
   metawhist[1] = 0.25;
   double pref = metahhist[1] / (metawhist[1] * std::sqrt(2.0 * pi));
   double sig2 = metawhist[1] * metawhist[1];
   double src[3] = {0.1, -0.1, 1.9};

   double lam[4] = {0.0, 0.1, 0.5, 1.0};
   for (int t = 0; t < 4; ++t) {
      CAPTURE(lam[t]);
      double vr = 0.0, dr = 0.0;
      for (int m = 0; m < 3; ++m) {
         double delta = lam[t] - src[m];
         double b = pref * std::exp(-0.5 * delta * delta / sig2);
         vr += b;
         dr -= delta * b / sig2;
      }
      eMetaBias(lam[t], vbias, dvdl);
      COMPARE_REALS(vbias, vr, 1.0e-12);
      COMPARE_REALS(dvdl, dr, 1.0e-12);
   }

   // right at the lambda = 0 wall the nearby image doubles the bias
   double single = pref * std::exp(-0.5 * 0.01 / sig2);
   double far = pref * std::exp(-0.5 * 1.9 * 1.9 / sig2);
   eMetaBias(0.0, vbias, dvdl);
   COMPARE_REALS(vbias, 2.0 * single + far, 1.0e-12);
   REQUIRE(vbias > single);
}

TEST_CASE("EOST-metatemper", "[ff][eost]")
{
   // drive eMetaDyn across several deposit intervals with tempering on.
   bath::kelvin = 300.0;
   resetost(5, 5, 1);
   resetmeta(8);
   iosthist = 4;
   ostnequil = 2;
   ostnavg = 2;
   hbias = 2.0;
   ostdedl = 0.0;
   ostdt = 0.0; // no-op ostLangevin, so the sampled lambda values stay controlled
   ostemper = true;
   temperthresh = 0.5;
   tempergamma = 1.0;

   // V* over the deposits already stored, summed independently of vmetagrid
   auto refVstar = [&](int upto) {
      double vmin = 1.0e30;
      for (int il = 1; il <= nlmda; ++il) {
         double lambda = (double)(il - 1) * wlmda;
         double v = 0.0;
         for (int j = 1; j <= upto; ++j) {
            double sig = metawhist[j];
            double sig2 = sig * sig;
            double pref = metahhist[j] / (sig * std::sqrt(2.0 * pi));
            double src[3] = {metalhist[j], -metalhist[j], 2.0 - metalhist[j]};
            for (int m = 0; m < 3; ++m) {
               double delta = lambda - src[m];
               v += pref * std::exp(-0.5 * delta * delta / sig2);
            }
         }
         vmin = std::min(vmin, v);
      }
      return vmin;
   };

   const int ndep = 5;
   for (int istep = 1; istep <= ndep * iosthist; ++istep) {
      ostlambda = 0.5;
      eMetaDyn(istep);
   }
   COMPARE_INTS(nmetahist, ndep);

   // the first deposit sees an empty bias, so it is untempered
   COMPARE_REALS(metahhist[1], hbias, 1.0e-12);
   REQUIRE(refVstar(1) > temperthresh); // the threshold really is crossed

   // every later height follows the rule applied to the pre-deposit V*, and the
   // sequence decays monotonically
   for (int k = 2; k <= ndep; ++k) {
      CAPTURE(k);
      COMPARE_REALS(metahhist[k], temperedHeight(refVstar(k - 1)), 1.0e-12);
      REQUIRE(metahhist[k] < metahhist[k - 1]);
   }

   // the accumulated grid matches the direct sum at every bin center, and the
   // Hermite evaluation reproduces it exactly at those nodes
   for (int il = 1; il <= nlmda; ++il) {
      CAPTURE(il);
      double lambda = (double)(il - 1) * wlmda;
      double vd, dd, vi, di;
      ostinterpol = false;
      eMetaBias(lambda, vd, dd);
      COMPARE_REALS(vmetagrid[il], vd, 1.0e-12);
      COMPARE_REALS(dvmetagrid[il], dd, 1.0e-12);
      ostinterpol = true;
      eMetaBias(lambda, vi, di);
      COMPARE_REALS(vi, vd, 1.0e-12);
      COMPARE_REALS(di, dd, 1.0e-12);
   }
   ostinterpol = false;
}
