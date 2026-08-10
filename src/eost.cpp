#include "ff/ost.h"
#include "ff/eost.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "math/const.h"
#include "math/random.h"
#include "tool/darray.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/ost.hh>
#include <tinker/detail/units.hh>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tinker {
// saved gaussian history (1-based, element 0 unused)
std::vector<int> osthist;
std::vector<int> ostihist; // iost/step stamp per deposited OST gaussian
std::vector<int> ostnext;
std::vector<int> osthead; // (nlmda x nflmda), column-major
std::vector<double> ostlhist, ostfhist, osthhist, ostwlhist, ostwfhist;

// per-step sample ring buffers, size iosthist+1, indexed 1..iosthist
std::vector<double> ostllist, ostflist;

// bias grids (nlmda x nflmda), column-major
std::vector<double> gkernel, glkernel, gfkernel, glfkernel;

// free-energy mean force per lambda bin, size nlmda+1, indexed 1..nlmda
std::vector<double> fkernel, fsumkernel, pfkernel;

// running max of gkernel over the flambda axis, size nlmda+1, indexed 1..nlmda
std::vector<double> vkernelmax;

// metadynamics gaussian history (1-based)
std::vector<double> metalhist, metahhist, metawhist;
std::vector<int> metaihist; // iost/step stamp per deposited metadynamics gaussian

// metadynamics bias and dV/dlambda at each lambda bin center, size nlmda+1
std::vector<double> vmetagrid, dvmetagrid;

// bias evaluated by eostBias
double bgbias, bdgdl, bdgdfl, bostlmda, bdfdl;

void ost_mech()
{
   ostinterpol = (ost::ostinterpol != 0);
   fastkernel = (ost::fastkernel != 0);

   iost = ost::iost;
   iosthist = ost::iosthist;
   ostnequil = ost::ostnequil;
   ostnavg = ost::ostnavg;
   nlmda = ost::nlmda;
   nflmda = ost::nflmda;
   fli0 = ost::fli0;
   nosthist = 0;
   sizeosthist = 0;
   nmetahist = 0;
   sizemetahist = 0;

   wlmda = ost::wlmda;
   wlmda2 = ost::wlmda2;
   wflmda = ost::wflmda;
   wflmda2 = ost::wflmda2;
   wlhist = ost::wlhist;
   wfhist = ost::wfhist;
   maxwlhist = ost::maxwlhist;
   maxwfhist = ost::maxwfhist;
   hbias = ost::hbias;
   oststdev = ost::oststdev;
   osteqratio = ost::osteqratio;
   ostcvbin = ost::ostcvbin;
   ostcvdif = ost::ostcvdif;
   ostcvrat = ost::ostcvrat;
   ostcvslp = ost::ostcvslp;
   ostcvstd = ost::ostcvstd;

   ostemper = (ost::ostemper != 0);
   temperthresh = ost::temperthresh;
   tempergamma = ost::tempergamma;

   osttheta = ost::osttheta;
   ostvtheta = ost::ostvtheta;
   ostmass = ost::ostmass;
   ostfriction = ost::ostfriction;
   ostdt = ost::ostdt;

   // dedl is owned by the energy routines and zeroed by zeroEGV each step.
   ostdgdl = 0;
   ostddgdl = 0;
   deffdl = 0;
   ostlambdaavg = 0;
   ostlambdastd = 0;
   ostlambdaslp = 0;
   ostdedlavg = 0;
   ostdedlstd = 0;
   ostdedlslp = 0;

   eosttot = 0;
}

static double fitSlope(double tdot, double sum, int n)
{
   if (n < 2)
      return 0.0;
   double sxx = (double)n * ((double)n * (double)n - 1.0) / 12.0;
   double sxy = tdot - 0.5 * (double)(n - 1) * sum;
   return sxy / sxx;
}

void histstat(const std::vector<double>& list, double& avg, double& std, double& slp,
   std::vector<double>& avgbin, std::vector<double>& stdbin, std::vector<double>& slpbin)
{
   int nper = (ostcvbin > 0) ? ostnavg / ostcvbin : 0;
   int nbin = (nper > 0) ? ostcvbin : 0;

   int ibegin = ostnequil + ostnavg - nper * nbin;
   if (ostcvbin > 0) {
      avgbin.assign(ostcvbin, 0.0);
      stdbin.assign(ostcvbin, 0.0);
      slpbin.assign(ostcvbin, 0.0);
   }

   const double K = list[ostnequil];
   double total = 0.0, tdot = 0.0;
   for (int i = ostnequil; i < ibegin; ++i) {
      double d = list[i] - K;
      total += d;
      tdot += (double)(i - ostnequil) * d;
   }
   for (int b = 0; b < nbin; ++b) {
      int i0 = ibegin + b * nper;
      double a = 0.0, tloc = 0.0;
      for (int i = i0; i < i0 + nper; ++i) {
         double d = list[i] - K;
         a += d;
         tloc += (double)(i - i0) * d;
      }
      total += a;
      // tloc counts from the bin start; shift it onto the whole-slice ramp
      tdot += tloc + (double)(i0 - ostnequil) * a;
      avgstd(list, i0, nper, avgbin[b], stdbin[b]);
      slpbin[b] = fitSlope(tloc, a, nper);
   }
   avgstd(list, ostnequil, ostnavg, avg, std);
   slp = fitSlope(tdot, total, ostnavg);
}

bool depcriteria(double avg, double std, double slp, const std::vector<double>& avgbin)
{
   if (std > ostcvstd)
      return false;
   if ((avg == 0.0 && std != 0.0) || (avg != 0.0 && std::abs(std / avg) > ostcvrat))
      return false;
   if (std::abs(slp) > ostcvslp)
      return false;
   if (avgbin.size() >= 2 && std::abs(avgbin.back() - avgbin.front()) > ostcvdif)
      return false;
   return true;
}

bool depcriteria2(double avg, double std)
{
   double tolerance = ostcvstd + ostcvrat * std::abs(avg);
   return tolerance > 0.0 && std / tolerance < 1.0;
}

double ostVminimax()
{
   // vkernelmax is 1-based with element 0 unused, so reduce over [1, nlmda].
   if (nlmda < 1 || (int)vkernelmax.size() < nlmda + 1)
      return 0.0;
   return *std::min_element(vkernelmax.begin() + 1, vkernelmax.begin() + nlmda + 1);
}

double metaVminimax()
{
   if (nlmda < 1 || (int)vmetagrid.size() < nlmda + 1)
      return 0.0;
   return *std::min_element(vmetagrid.begin() + 1, vmetagrid.begin() + nlmda + 1);
}

// h = hbias * exp(-max(0, vminimax - temperthresh) / (kT * tempergamma))
double temperedHeight(double vminimax)
{
   if (not ostemper)
      return hbias;
   const double rt = units::gasconst * bath::kelvin;
   double denom = rt * tempergamma;
   if (denom <= 0.0)
      return hbias;
   double excess = std::max(0.0, vminimax - temperthresh);
   return hbias * std::exp(-excess / denom);
}

// buildostindex -- rebuild the packed bins and linked-list lookup from the saved
// real centers (eost.f:1240).
void buildOstIndex()
{
   std::fill(osthead.begin(), osthead.end(), 0);
   for (int i = 1; i <= sizeosthist; ++i)
      ostnext[i] = 0;
   for (int ihist = 1; ihist <= nosthist; ++ihist) {
      int il = lambdaBin(ostlhist[ihist]);
      int jf = flambdaBin(ostfhist[ihist]);
      int k;
      ijToK(il, jf, nlmda, k);
      osthist[ihist] = k;
      ostnext[ihist] = osthead[gidx(il, jf)];
      osthead[gidx(il, jf)] = ihist;
   }
}

// resizeosthist -- double the history storage, preserving saved data
// (eost.f:1144). std::vector::resize keeps existing elements.
void resizeOstHist()
{
   int newsize = 2 * sizeosthist;
   sizeosthist = newsize;
   osthist.resize(newsize + 1, 0);
   ostihist.resize(newsize + 1, 0);
   ostnext.resize(newsize + 1, 0);
   ostlhist.resize(newsize + 1, 0.0);
   ostfhist.resize(newsize + 1, 0.0);
   osthhist.resize(newsize + 1, 0.0);
   ostwlhist.resize(newsize + 1, 0.0);
   ostwfhist.resize(newsize + 1, 0.0);
}

// ensureflambda -- grow the flambda grid (and shift/reallocate the four kernel
// grids) if the new dU/dlambda falls too near the current edge (eost.f:490).
void ensureFlambda(double dudl)
{
   int iflmda = (int)std::lround(dudl / wflmda) + fli0;

   int nchunk = 100;
   int nbuffer = nchunk;
   int nfcut = (int)(oststdev * maxwfhist / wflmda);
   if ((double)nfcut * wflmda < oststdev * maxwfhist + wflmda2)
      nfcut = nfcut + 1;
   nfcut = nfcut + nbuffer;
   if (iflmda - nfcut >= 1 && iflmda + nfcut <= nflmda)
      return;

   int oldnflmda = nflmda;
   int oldfli0 = fli0;
   int naddlow = 0;
   int naddhigh = 0;
   if (iflmda - nfcut < 1) {
      int nneed = 1 - (iflmda - nfcut);
      naddlow = ((nneed + nchunk - 1) / nchunk) * nchunk;
      nflmda += naddlow;
      fli0 += naddlow;
      iflmda += naddlow;
   }
   if (iflmda + nfcut > nflmda) {
      int nneed = iflmda + nfcut - nflmda;
      naddhigh = ((nneed + nchunk - 1) / nchunk) * nchunk;
      nflmda += naddhigh;
   }
   (void)naddhigh;
   int offset = fli0 - oldfli0;

   // rebuild the lookup index at the new size
   osthead.assign((size_t)nlmda * nflmda, 0);
   buildOstIndex();

   // reallocate the flambda-dependent kernels, shifting old data by offset.
   std::vector<double> gf0 = gfkernel, g0 = gkernel, glf0 = glfkernel, gl0 = glkernel;
   gfkernel.assign((size_t)nlmda * nflmda, 0.0);
   gkernel.assign((size_t)nlmda * nflmda, 0.0);
   glfkernel.assign((size_t)nlmda * nflmda, 0.0);
   glkernel.assign((size_t)nlmda * nflmda, 0.0);
   for (int i = 1; i <= nlmda; ++i) {
      for (int j = 1; j <= oldnflmda; ++j) {
         int src = (i - 1) + (j - 1) * nlmda;
         int dst = (i - 1) + (j - 1 + offset) * nlmda;
         gfkernel[dst] = gf0[src];
         gkernel[dst] = g0[src];
         glfkernel[dst] = glf0[src];
         glkernel[dst] = gl0[src];
      }
   }
}

// addkernelpoint -- update one g kernel cell and the associated f kernel
// numerator/partition-function accumulators (eost.f:1606).
void addKernelPoint(int ilmda, int iflmda, double e, double ldelta, double fldelta, double sigl2, double sigf2)
{
   const double rt = units::gasconst * bath::kelvin;
   int g = gidx(ilmda, iflmda);
   double oldg = gkernel[g];
   double oldweight = (oldg == 0.0) ? 0.0 : std::exp(oldg / rt);
   double newg = oldg + e;
   double newweight = std::exp(newg / rt);
   double delweight = newweight - oldweight;
   double flmda = (double)(iflmda - fli0) * wflmda;
   double dgdl = -ldelta * e / sigl2;
   double dgdfl = -fldelta * e / sigf2;
   double d2gdlfl = ldelta * fldelta * e / (sigl2 * sigf2);
   gfkernel[g] += dgdfl;
   gkernel[g] = newg;
   vkernelmax[ilmda] = std::max(vkernelmax[ilmda], newg);
   glfkernel[g] += d2gdlfl;
   glkernel[g] += dgdl;
   fsumkernel[ilmda] += flmda * delweight;
   pfkernel[ilmda] += delweight;
   if (pfkernel[ilmda] == 0.0)
      fkernel[ilmda] = 0.0;
   else
      fkernel[ilmda] = fsumkernel[ilmda] / pfkernel[ilmda];
}

// Spread one saved histogram source over nearby grid bins. When do_f is true the
// f-kernel accumulators are updated too (addkernelhist); otherwise only the g
// kernel value is spread (addgkernelhist). eost.f:1409 / 1507.
void addKernelHistImpl(int ihist, bool do_f)
{
   int k = osthist[ihist];
   int lsrc, fsrc;
   kToIj(k, nlmda, lsrc, fsrc);
   double sigl = ostwlhist[ihist];
   double sigf = ostwfhist[ihist];
   double sigl2 = sigl * sigl;
   double sigf2 = sigf * sigf;
   double pref = osthhist[ihist] / (2.0 * pi * sigl * sigf);
   double sourcefl = ostfhist[ihist];

   int nlcut = (int)(oststdev * sigl / wlmda);
   if ((double)nlcut * wlmda < oststdev * sigl + wlmda2)
      nlcut = nlcut + 1;
   int nfcut = (int)(oststdev * sigf / wflmda);
   if ((double)nfcut * wflmda < oststdev * sigf + wflmda2)
      nfcut = nfcut + 1;
   int iflmda1 = std::max(1, fsrc - nfcut);
   int iflmda2 = std::min(nflmda, fsrc + nfcut);

   for (int img = 1; img <= 3; ++img) {
      int llog;
      double sourcel;
      if (img == 1) {
         llog = lsrc;
         sourcel = ostlhist[ihist];
      } else if (img == 2) {
         llog = 2 - lsrc;
         sourcel = -ostlhist[ihist];
      } else {
         llog = 2 * nlmda - lsrc;
         sourcel = 2.0 - ostlhist[ihist];
      }
      int ilmda1 = std::max(1, llog - nlcut);
      int ilmda2 = std::min(nlmda, llog + nlcut);
      if (ilmda1 > ilmda2)
         continue;
      for (int ilmda = ilmda1; ilmda <= ilmda2; ++ilmda) {
         double targetl = (double)(ilmda - 1) * wlmda;
         double ldelta = targetl - sourcel;
         if (std::fabs(ldelta) > oststdev * sigl)
            continue;
         double ldelta2 = ldelta * ldelta;
         double expl = std::exp(-ldelta2 / (2.0 * sigl2));
         for (int iflmda = iflmda1; iflmda <= iflmda2; ++iflmda) {
            double targetf = (double)(iflmda - fli0) * wflmda;
            double fldelta = targetf - sourcefl;
            if (std::fabs(fldelta) > oststdev * sigf)
               continue;
            double fldelta2 = fldelta * fldelta;
            double expfl = std::exp(-fldelta2 / (2.0 * sigf2));
            double e = pref * expl * expfl;
            if (do_f) {
               addKernelPoint(ilmda, iflmda, e, ldelta, fldelta, sigl2, sigf2);
            } else {
               double newg = gkernel[gidx(ilmda, iflmda)] + e;
               gkernel[gidx(ilmda, iflmda)] = newg;
               vkernelmax[ilmda] = std::max(vkernelmax[ilmda], newg);
            }
         }
      }
   }
}

// buildfkernel -- rebuild the mean force at every lambda bin from the g kernel
// (eost.f:1664).
void buildFkernel()
{
   const double rt = units::gasconst * bath::kelvin;
   for (int il = 1; il <= nlmda; ++il) {
      double avg = 0;
      double pf = 0;
      for (int jf = 1; jf <= nflmda; ++jf) {
         double g = gkernel[gidx(il, jf)];
         if (g != 0.0) {
            double flmda = (double)(jf - fli0) * wflmda;
            double w = std::exp(g / rt);
            avg += flmda * w;
            pf += w;
         }
      }
      fkernel[il] = (pf == 0.0) ? 0.0 : avg / pf;
   }
}

// egkernel -- direct linked-list gaussian sum of the g bias and its partials at
// the current (lambda, dedl) (eost.f:851).
void egkernel(double& egbias, double& dgdl, double& dgdfl)
{
   egbias = 0;
   dgdl = 0;
   dgdfl = 0;

   int ilmda = lambdaBin(lambda);
   int iflmda = (int)std::lround(dedl / wflmda) + fli0;
   if (iflmda < 1 || iflmda > nflmda)
      return;

   int nlcut = (int)(oststdev * maxwlhist / wlmda);
   if ((double)nlcut * wlmda < oststdev * maxwlhist)
      nlcut = nlcut + 1;
   nlcut = nlcut + 1;
   int nfcut = (int)(oststdev * maxwfhist / wflmda);
   if ((double)nfcut * wflmda < oststdev * maxwfhist)
      nfcut = nfcut + 1;
   nfcut = nfcut + 1;

   for (int klmda = -nlcut; klmda <= nlcut; ++klmda) {
      int lcenter = ilmda + klmda;
      int lcount = lcenter;
      if (lcount < 1)
         lcount = 2 - lcount;
      else if (lcount > nlmda)
         lcount = 2 * nlmda - lcount;
      if (lcount < 1 || lcount > nlmda)
         continue;
      for (int kflmda = -nfcut; kflmda <= nfcut; ++kflmda) {
         int flcenter = iflmda + kflmda;
         if (flcenter < 1 || flcenter > nflmda)
            continue;
         int ihist = osthead[gidx(lcount, flcenter)];
         while (ihist != 0) {
            double sigl = ostwlhist[ihist];
            double sigf = ostwfhist[ihist];
            double sourcefl = ostfhist[ihist];
            double fldelta = dedl - sourcefl;
            if (std::fabs(fldelta) <= oststdev * sigf) {
               double sigl2 = sigl * sigl;
               double sigf2 = sigf * sigf;
               double sigl2inv = 1.0 / sigl2;
               double sigf2inv = 1.0 / sigf2;
               double pref = osthhist[ihist] / (2.0 * pi * sigl * sigf);
               double fldelta2 = fldelta * fldelta;
               double expfl = std::exp(-0.5 * fldelta2 * sigf2inv);
               int nimg = 1;
               if (lcenter == 1 || lcenter == nlmda)
                  nimg = 2;
               for (int img = 1; img <= nimg; ++img) {
                  double sourcel;
                  if (lcenter < 1)
                     sourcel = -ostlhist[ihist];
                  else if (lcenter > nlmda)
                     sourcel = 2.0 - ostlhist[ihist];
                  else if (img == 2 && lcenter == 1)
                     sourcel = -ostlhist[ihist];
                  else if (img == 2 && lcenter == nlmda)
                     sourcel = 2.0 - ostlhist[ihist];
                  else
                     sourcel = ostlhist[ihist];
                  double ldelta = lambda - sourcel;
                  if (std::fabs(ldelta) <= oststdev * sigl) {
                     double ldelta2 = ldelta * ldelta;
                     double expl = std::exp(-0.5 * ldelta2 * sigl2inv);
                     double bias = pref * expl * expfl;
                     egbias += bias;
                     dgdl -= ldelta * sigl2inv * bias;
                     dgdfl -= fldelta * sigf2inv * bias;
                  }
               }
            }
            ihist = ostnext[ihist];
         }
      }
   }
}

// egkernelinterpolate -- bicubic Hermite evaluation of the g kernel and its first
// derivatives from the grids (eost.f:979).
void egkernelInterpolate(double& egbias, double& dgdl, double& dgdfl)
{
   egbias = 0;
   dgdl = 0;
   dgdfl = 0;
   double flstart = (double)(1 - fli0) * wflmda;
   double flend = (double)(nflmda - fli0) * wflmda;
   if (lambda < 0.0 || lambda > 1.0)
      return;
   if (dedl < flstart || dedl > flend)
      return;

   int il0;
   if (lambda >= 1.0)
      il0 = nlmda - 1;
   else {
      il0 = (int)(lambda / wlmda) + 1;
      il0 = std::max(1, std::min(il0, nlmda - 1));
   }
   int if0;
   if (dedl >= flend)
      if0 = nflmda - 1;
   else {
      if0 = (int)((dedl - flstart) / wflmda) + 1;
      if0 = std::max(1, std::min(if0, nflmda - 1));
   }
   double l0 = (double)(il0 - 1) * wlmda;
   double f0 = (double)(if0 - fli0) * wflmda;
   double x = (lambda - l0) / wlmda;
   double y = (dedl - f0) / wflmda;

   double x2 = x * x, x3 = x2 * x;
   double y2 = y * y, y3 = y2 * y;
   double hxv[2] = {2.0 * x3 - 3.0 * x2 + 1.0, -2.0 * x3 + 3.0 * x2};
   double hxd[2] = {x3 - 2.0 * x2 + x, x3 - x2};
   double dhxv[2] = {6.0 * x2 - 6.0 * x, -6.0 * x2 + 6.0 * x};
   double dhxd[2] = {3.0 * x2 - 4.0 * x + 1.0, 3.0 * x2 - 2.0 * x};
   double hyv[2] = {2.0 * y3 - 3.0 * y2 + 1.0, -2.0 * y3 + 3.0 * y2};
   double hyd[2] = {y3 - 2.0 * y2 + y, y3 - y2};
   double dhyv[2] = {6.0 * y2 - 6.0 * y, -6.0 * y2 + 6.0 * y};
   double dhyd[2] = {3.0 * y2 - 4.0 * y + 1.0, 3.0 * y2 - 2.0 * y};

   for (int ia = 1; ia <= 2; ++ia) {
      int i = il0 + ia - 1;
      for (int ja = 1; ja <= 2; ++ja) {
         int j = if0 + ja - 1;
         double val = gkernel[gidx(i, j)];
         double gl = wlmda * glkernel[gidx(i, j)];
         double gf = wflmda * gfkernel[gidx(i, j)];
         double glf = wlmda * wflmda * glfkernel[gidx(i, j)];
         egbias += hxv[ia - 1] * hyv[ja - 1] * val + hxd[ia - 1] * hyv[ja - 1] * gl
            + hxv[ia - 1] * hyd[ja - 1] * gf + hxd[ia - 1] * hyd[ja - 1] * glf;
         dgdl += dhxv[ia - 1] * hyv[ja - 1] * val + dhxd[ia - 1] * hyv[ja - 1] * gl
            + dhxv[ia - 1] * hyd[ja - 1] * gf + dhxd[ia - 1] * hyd[ja - 1] * glf;
         dgdfl += hxv[ia - 1] * dhyv[ja - 1] * val + hxd[ia - 1] * dhyv[ja - 1] * gl
            + hxv[ia - 1] * dhyd[ja - 1] * gf + hxd[ia - 1] * dhyd[ja - 1] * glf;
      }
   }
   dgdl /= wlmda;
   dgdfl /= wflmda;
}

// efkernel -- DeltaG(lambda) and dDeltaG/dlambda by piecewise-linear
// integration of the mean force (eost.f:1750).
void efkernel(double& eostlmda, double& dfdl)
{
   eostlmda = 0;
   dfdl = 0;
   if (lambda <= 0.0) {
      dfdl = fkernel[1];
      return;
   }
   for (int il0 = 1; il0 <= nlmda - 1; ++il0) {
      int il1 = il0 + 1;
      double lmda0 = (double)(il0 - 1) * wlmda;
      double lmda1 = (double)(il1 - 1) * wlmda;
      double fl0 = fkernel[il0];
      double fl1 = fkernel[il1];
      double slope = (fl1 - fl0) / wlmda;
      if (lambda <= lmda1) {
         double xx = lambda - lmda0;
         eostlmda += fl0 * xx + 0.5 * slope * xx * xx;
         dfdl = fl0 + slope * xx;
         return;
      }
      eostlmda += 0.5 * (fl0 + fl1) * wlmda;
   }
   dfdl = fkernel[nlmda];
}

// etotfkernel -- total DeltaG by trapezoid integration of the mean force
// (eost.f:1717).
double etotFkernel()
{
   double tot = 0;
   for (int il = 1; il <= nlmda - 1; ++il)
      tot += 0.5 * (fkernel[il] + fkernel[il + 1]) * wlmda;
   return tot;
}

// ostlangevin -- BAOAB Langevin propagation of the theta lambda-particle where
// lambda = sin(theta)^2 (eost.f:374).
void ostLangevin()
{
   if (ostdt <= 0.0)
      return;
   if (ostmass <= 0.0)
      return;

   double force = -deffdl * std::sin(2.0 * osttheta);
   double gamma = std::max(0.0, ostfriction);
   if (gamma > 0.0) {
      double c = std::exp(-gamma * ostdt);
      double ktm = units::boltzmann * bath::kelvin / ostmass;
      double sigma = std::sqrt(ktm * (1.0 - c * c));
      ostvtheta = c * ostvtheta + (1.0 - c) * force / (gamma * ostmass) + sigma * normal<double>();
   } else {
      ostvtheta = ostvtheta + ostdt * force / ostmass;
   }

   osttheta = osttheta + ostdt * ostvtheta;
   while (osttheta > pi)
      osttheta -= 2.0 * pi;
   while (osttheta <= -pi)
      osttheta += 2.0 * pi;

   double sinth = std::sin(osttheta);
   lambda = sinth * sinth;
}

static void metaImages(double lmda, double src[3])
{
   src[0] = lmda;
   src[1] = -lmda;
   src[2] = 2.0 - lmda;
}

// emetabias -- Vbias(lambda) and dVbias/dlambda for the sum of 1D normalized
// metadynamics gaussians (eost.f:244).
void eMetaBias(double lmda, double& vbias, double& dvdl)
{
   vbias = 0;
   dvdl = 0;

   if (ostinterpol && nmetahist > 0) {
      eMetaBiasInterpolate(lmda, vbias, dvdl);
      return;
   }

   for (int ihist = 1; ihist <= nmetahist; ++ihist) {
      double sig = metawhist[ihist];
      if (sig > 0.0) {
         double sig2 = sig * sig;
         double pref = metahhist[ihist] / (sig * std::sqrt(2.0 * pi));
         double src[3];
         metaImages(metalhist[ihist], src);
         for (int img = 0; img < 3; ++img) {
            double delta = lmda - src[img];
            double bias = pref * std::exp(-0.5 * delta * delta / sig2);
            vbias += bias;
            dvdl -= delta * bias / sig2;
         }
      }
   }
}

// addmetagrid -- accumulate one deposited metadynamics gaussian
void addMetaGrid(int ihist)
{
   double sig = metawhist[ihist];
   if (sig <= 0.0)
      return;
   if ((int)vmetagrid.size() < nlmda + 1 || (int)dvmetagrid.size() < nlmda + 1)
      return;
   double sig2 = sig * sig;
   double pref = metahhist[ihist] / (sig * std::sqrt(2.0 * pi));
   double src[3];
   metaImages(metalhist[ihist], src);
   for (int il = 1; il <= nlmda; ++il) {
      double lmda = (double)(il - 1) * wlmda;
      for (int img = 0; img < 3; ++img) {
         double delta = lmda - src[img];
         double bias = pref * std::exp(-0.5 * delta * delta / sig2);
         vmetagrid[il] += bias;
         dvmetagrid[il] -= delta * bias / sig2;
      }
   }
}

// emetabiasinterpolate -- cubic Hermite evaluation of the metadynamics bias.
void eMetaBiasInterpolate(double lmda, double& vbias, double& dvdl)
{
   vbias = 0;
   dvdl = 0;
   if (nlmda < 2)
      return;
   if ((int)vmetagrid.size() < nlmda + 1 || (int)dvmetagrid.size() < nlmda + 1)
      return;

   double lam = std::min(1.0, std::max(0.0, lmda));
   int il0;
   if (lam >= 1.0) {
      il0 = nlmda - 1;
   } else {
      il0 = (int)(lam / wlmda) + 1;
      il0 = std::max(1, std::min(il0, nlmda - 1));
   }
   double l0 = (double)(il0 - 1) * wlmda;
   double x = (lam - l0) / wlmda;

   double x2 = x * x, x3 = x2 * x;
   double hxv[2] = {2.0 * x3 - 3.0 * x2 + 1.0, -2.0 * x3 + 3.0 * x2};
   double hxd[2] = {x3 - 2.0 * x2 + x, x3 - x2};
   double dhxv[2] = {6.0 * x2 - 6.0 * x, -6.0 * x2 + 6.0 * x};
   double dhxd[2] = {3.0 * x2 - 4.0 * x + 1.0, 3.0 * x2 - 2.0 * x};

   for (int ia = 0; ia < 2; ++ia) {
      int i = il0 + ia;
      double val = vmetagrid[i];
      double der = wlmda * dvmetagrid[i];
      vbias += hxv[ia] * val + hxd[ia] * der;
      dvdl += dhxv[ia] * val + dhxd[ia] * der;
   }
   dvdl /= wlmda;
}

// metadeltag -- G(1) - G(0) = -Vbias(1) + Vbias(0) (eost.f:288).
double metaDeltaG()
{
   double v0, v1, dvdl;
   eMetaBias(0.0, v0, dvdl);
   eMetaBias(1.0, v1, dvdl);
   return -v1 + v0;
}

void resizeMeta()
{
   int newsize = 2 * sizemetahist;
   sizemetahist = newsize;
   metalhist.resize(newsize + 1, 0.0);
   metahhist.resize(newsize + 1, 0.0);
   metawhist.resize(newsize + 1, 0.0);
   metaihist.resize(newsize + 1, 0);
}

// Device scaffold (disabled). A future GPU port of the bicubic-Hermite bias
// evaluator would be dispatched here, following the adtMix pattern in ost.cpp.
// See src/acc/eost.cpp and src/cu/eost.cu. Enable together with those files.
//   TINKER_FVOID2(acc0, cu1, eostBiasGrid, /* grid ptrs, sample, outputs */);

// addgkernelhist / addkernelhist -- spread one saved source over the grid; the
// merged worker is addKernelHistImpl (eost.f:1409 / 1507).
void addGkernelHist(int ihist)
{
   addKernelHistImpl(ihist, false);
}

void addKernelHist(int ihist)
{
   addKernelHistImpl(ihist, true);
}

// buildgkernel -- clear stale values and rebuild the g kernel from scratch over
// all saved sources (eost.f:1287).
void buildGkernel()
{
   std::fill(gkernel.begin(), gkernel.end(), 0.0);
   std::fill(vkernelmax.begin(), vkernelmax.end(), 0.0);
   for (int ihist = 1; ihist <= nosthist; ++ihist)
      addGkernelHist(ihist);
}

// buildkernels -- clear and rebuild both kernels and the free-energy
// accumulators from scratch over all saved sources (eost.f:1322).
void buildKernels()
{
   std::fill(gkernel.begin(), gkernel.end(), 0.0);
   std::fill(gfkernel.begin(), gfkernel.end(), 0.0);
   std::fill(glfkernel.begin(), glfkernel.end(), 0.0);
   std::fill(glkernel.begin(), glkernel.end(), 0.0);
   std::fill(fkernel.begin(), fkernel.end(), 0.0);
   std::fill(fsumkernel.begin(), fsumkernel.end(), 0.0);
   std::fill(pfkernel.begin(), pfkernel.end(), 0.0);
   std::fill(vkernelmax.begin(), vkernelmax.end(), 0.0);
   for (int ihist = 1; ihist <= nosthist; ++ihist)
      addKernelHist(ihist);
}

// updategkernel / updatekernels -- spread only the most recently saved source
// into the current grids (eost.f:1363 / 1386).
void updateGkernel()
{
   if (nosthist > 0)
      addGkernelHist(nosthist);
}

void updateKernels()
{
   if (nosthist > 0)
      addKernelHist(nosthist);
}

void eostData(RcOp op)
{
   if (!(use_ost || use_meta))
      return;

   if (op & RcOp::DEALLOC) {
      osthist.clear();
      ostihist.clear();
      ostnext.clear();
      osthead.clear();
      ostlhist.clear();
      ostfhist.clear();
      osthhist.clear();
      ostwlhist.clear();
      ostwfhist.clear();
      ostllist.clear();
      ostflist.clear();
      gkernel.clear();
      glkernel.clear();
      gfkernel.clear();
      glfkernel.clear();
      fkernel.clear();
      fsumkernel.clear();
      pfkernel.clear();
      vkernelmax.clear();
      metalhist.clear();
      metahhist.clear();
      metawhist.clear();
      metaihist.clear();
      vmetagrid.clear();
      dvmetagrid.clear();
      ostlambdaavgbin.clear();
      ostlambdastdbin.clear();
      ostlambdaslpbin.clear();
      ostdedlavgbin.clear();
      ostdedlstdbin.clear();
      ostdedlslpbin.clear();
   }

   if (op & RcOp::INIT) {
      bgbias = 0;
      bdgdl = 0;
      bdgdfl = 0;
      bostlmda = 0;
      bdfdl = 0;

      int ncvbin = std::max(ostcvbin, 0);
      ostlambdaavgbin.assign(ncvbin, 0.0);
      ostlambdastdbin.assign(ncvbin, 0.0);
      ostlambdaslpbin.assign(ncvbin, 0.0);
      ostdedlavgbin.assign(ncvbin, 0.0);
      ostdedlstdbin.assign(ncvbin, 0.0);
      ostdedlslpbin.assign(ncvbin, 0.0);

      // Mirror the Fortran mutate allocation/initialization (mutate.f:505).
      if (use_ost) {
         sizeosthist = 10000;
         nosthist = 0;
         osthist.assign(sizeosthist + 1, 0);
         ostihist.assign(sizeosthist + 1, 0);
         ostnext.assign(sizeosthist + 1, 0);
         osthead.assign((size_t)nlmda * nflmda, 0);
         ostllist.assign(iosthist, 0.0);
         ostflist.assign(iosthist, 0.0);
         ostlhist.assign(sizeosthist + 1, 0.0);
         ostfhist.assign(sizeosthist + 1, 0.0);
         osthhist.assign(sizeosthist + 1, 0.0);
         ostwlhist.assign(sizeosthist + 1, 0.0);
         ostwfhist.assign(sizeosthist + 1, 0.0);
         fkernel.assign(nlmda + 1, 0.0);
         fsumkernel.assign(nlmda + 1, 0.0);
         pfkernel.assign(nlmda + 1, 0.0);
         vkernelmax.assign(nlmda + 1, 0.0);
         gkernel.assign((size_t)nlmda * nflmda, 0.0);
         gfkernel.assign((size_t)nlmda * nflmda, 0.0);
         glfkernel.assign((size_t)nlmda * nflmda, 0.0);
         glkernel.assign((size_t)nlmda * nflmda, 0.0);
      }
      if (use_meta) {
         sizemetahist = 10000;
         nmetahist = 0;
         metalhist.assign(sizemetahist + 1, 0.0);
         metahhist.assign(sizemetahist + 1, 0.0);
         metawhist.assign(sizemetahist + 1, 0.0);
         metaihist.assign(sizemetahist + 1, 0);
         ostllist.assign(iosthist, 0.0);
         vmetagrid.assign(nlmda + 1, 0.0);
         dvmetagrid.assign(nlmda + 1, 0.0);
      }
   }
}

void eostBias(int vers)
{
   // dedl is the unbiased dU/dlambda for this configuration, accumulated by the
   // energy terms and chain ruled by lmdachain just above. Nothing below writes
   // it, so it stays valid until eostDyn/eMetaDyn consume it after energy()
   // returns; zeroEGV zeroes it again at the top of the next evaluation.
   if (use_meta) {
      eMetaBias(lambda, bgbias, bdgdl);
      // Vbias depends on lambda alone, so it carries no Cartesian force/virial.
      if (vers & calc::energy)
         esum += bgbias;
      return;
   }
   if (not use_ost)
      return;

   // Evaluate the g bias and the f (free-energy) term at the current state.
   if (ostinterpol)
      egkernelInterpolate(bgbias, bdgdl, bdgdfl);
   else
      egkernel(bgbias, bdgdl, bdgdfl);
   efkernel(bostlmda, bdfdl);

   if (vers & calc::energy)
      esum += bgbias - bostlmda;

   // The g bias depends on dU/dlambda, whose gradient/virial are dfsumdl* and
   // dvirdl, so these are the OSRW second-order Cartesian terms. The force term
   // runs on device via sumGradient.
   if ((vers & calc::grad) && dfsumdlx)
      sumGradient(bdgdfl, gx, gy, gz, dfsumdlx, dfsumdly, dfsumdlz);
   if (vers & calc::virial)
      for (int k = 0; k < 9; ++k)
         vir[k] += bdgdfl * dvirdl[k];
}

void eostDyn(int istep)
{
   int im = istep % iosthist;
   int isamp = (istep - 1) % iosthist;

   // effective lambda force, from the bias eostBias evaluated this step and the
   // unbiased dedl left behind by the energy call.
   ostdgdl = bdgdl + bdgdfl * d2edl2;
   ostddgdl = bdfdl;
   deffdl = dedl + ostdgdl - ostddgdl;

   // buffer this step's sample.
   ostllist[isamp] = lambda;
   ostflist[isamp] = dedl;

   // deposit a new histogram gaussian every iosthist steps.
   if (im == 0) {
      histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
         ostlambdaslpbin);
      histstat(ostflist, ostdedlavg, ostdedlstd, ostdedlslp, ostdedlavgbin, ostdedlstdbin, ostdedlslpbin);
      // if (depcriteria(ostdedlavg, ostdedlstd, ostdedlslp, ostdedlavgbin)) {
      if (depcriteria2(ostdedlavg, ostdedlstd)) {
         int ilmda = lambdaBin(ostlambdaavg);
         maxwlhist = std::max(maxwlhist, wlhist);
         maxwfhist = std::max(maxwfhist, wfhist);
         ensureFlambda(ostdedlavg);
         int iflmda = flambdaBin(ostdedlavg);

         nosthist = nosthist + 1;
         if (nosthist > sizeosthist)
            resizeOstHist();
         int k;
         ijToK(ilmda, iflmda, nlmda, k);
         osthist[nosthist] = k;
         ostihist[nosthist] = istep;
         ostlhist[nosthist] = ostlambdaavg;
         ostfhist[nosthist] = ostdedlavg;
         osthhist[nosthist] = temperedHeight(ostVminimax());
         ostwlhist[nosthist] = wlhist;
         ostwfhist[nosthist] = wfhist;
         ostnext[nosthist] = osthead[gidx(ilmda, iflmda)];
         osthead[gidx(ilmda, iflmda)] = nosthist;

         if (true) {
            double vmm = ostVminimax();
            double th = temperedHeight(vmm);
            printf("istep: %i\n", istep);
            printf("ostlmda  avg, std, slp: %8.4f %8.4e %8.4e\n", ostlambdaavg, ostlambdastd, ostlambdaslp);
            printf("lmda[0]  avg, std, slp: %8.4f %8.4e %8.4e\n", ostlambdaavgbin.front(), ostlambdastdbin.front(), ostlambdaslpbin.front());
            printf("lmda[-1] avg, std, slp: %8.4f %8.4e %8.4e\n", ostlambdaavgbin.back(), ostlambdastdbin.back(), ostlambdaslpbin.back());
            printf("ostdedl  avg, std, slp: %8.4f %8.4e %8.4e\n", ostdedlavg, ostdedlstd, ostdedlslp);
            printf("dedl[0]  avg, std, slp: %8.4f %8.4e %8.4e\n", ostdedlavgbin.front(), ostdedlstdbin.front(), ostdedlslpbin.front());
            printf("dedl[-1] avg, std, slp: %8.4f %8.4e %8.4e\n", ostdedlavgbin.back(), ostdedlstdbin.back(), ostdedlslpbin.back());
            printf("vminmax temperedHeight: %8.4e %8.4e\n", vmm, th);
            printf("\n");
         }

         if (fastkernel) {
            updateKernels();
         } else {
            updateGkernel();
            buildFkernel();
         }
         eosttot = etotFkernel();
      }
   }

   // propagate the lambda particle for the next step.
   ostLangevin();
}

void eMetaDyn(int istep)
{
   int im = istep % iosthist;
   int isamp = (istep - 1) % iosthist;

   // effective lambda force, from the bias eostBias evaluated this step and the
   // unbiased dedl left behind by the energy call.
   deffdl = dedl + bdgdl;

   // buffer this step's sample.
   ostllist[isamp] = lambda;

   // deposit a new metadynamics gaussian every iosthist steps.
   if (im == 0) {
      histstat(ostllist, ostlambdaavg, ostlambdastd, ostlambdaslp, ostlambdaavgbin, ostlambdastdbin,
         ostlambdaslpbin);
      nmetahist = nmetahist + 1;
      if (nmetahist > sizemetahist)
         resizeMeta();
      metalhist[nmetahist] = ostlambdaavg;
      metahhist[nmetahist] = temperedHeight(metaVminimax());
      metawhist[nmetahist] = wlmda;
      metaihist[nmetahist] = istep;
      addMetaGrid(nmetahist);
      eosttot = metaDeltaG();
   }

   ostLangevin();
}
}
