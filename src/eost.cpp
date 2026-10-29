#include "ff/ost.h"
#include "ff/eost.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "math/const.h"
#include "tool/darray.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/ost.hh>
#include <tinker/detail/units.hh>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tinker {
// the gaussian prefactors follow eost.f in double precision, and tinker::pi is
// only a float in a mixed precision build
static constexpr double dpi = M_PI;

// saved gaussian history (1-based, element 0 unused)
std::vector<int> osthist;
std::vector<int> ostnext;
std::vector<int> osthead; // (nlmda x nflmda), column-major
std::vector<double> osthhist, ostwlhist, ostwfhist;

// bias grids (nlmda x nflmda), column-major
std::vector<double> gkernel, glkernel, gfkernel, glfkernel;

// running max of gkernel over the flambda axis, size nlmda+1, indexed 1..nlmda
std::vector<double> vkernelmax;

// metadynamics gaussian history (1-based)
std::vector<double> metalhist, metahhist, metawhist;
std::vector<int> metaihist; // lmdastep stamp per deposited metadynamics gaussian
std::vector<int> metanext;  // next metadynamics gaussian in the same lambda bin
std::vector<int> metahead;  // first metadynamics gaussian of each lambda bin, size nlmda+1

// metadynamics bias and dV/dlambda at each lambda bin center, size nlmda+1
std::vector<double> vmetagrid, dvmetagrid;

// bias evaluated by eostBias
double bgbias, bdgdl, bdgdfl, bostlmda;

void ost_mech()
{
   ostinterpol = (ost::ostinterpol != 0);
   fastkernel = (ost::fastkernel != 0);

   nflmda = ost::nflmda;
   fli0 = ost::fli0;
   nmetahist = 0;
   sizemetahist = 0;

   wflmda = ost::wflmda;
   wflmda2 = ost::wflmda2;
   wlhist = ost::wlhist;
   wfhist = ost::wfhist;
   maxwlhist = ost::maxwlhist;
   maxwfhist = ost::maxwfhist;
   hbias = ost::hbias;
   oststdev = ost::oststdev;

   use_ostgtemp = (ost::use_ostgtemp != 0);
   use_ostltemp = (ost::use_ostltemp != 0);
   ostgthresh = ost::ostgthresh;
   ostgtempgamma = ost::ostgtempgamma;
   ostlthresh = ost::ostlthresh;
   ostltempgamma = ost::ostltempgamma;

   // dedl is owned by the energy routines and zeroed by zeroEGV each step.
   ostdgdl = 0;
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

// h = hbias * exp(-max(0, vglobal - ostgthresh) / (kT * ostgtempgamma)
//                 -max(0, vlocal - vglobal - ostlthresh) / (kT * ostltempgamma))
double temperedHeight(double vglobal, double vlocal)
{
   const double rt = units::gasconst * bath::kelvin;
   if (rt <= 0.0)
      return hbias;
   double expo = 0.0;
   // the global factor follows the path bias level
   if (use_ostgtemp and ostgtempgamma > 0.0)
      expo -= std::max(0.0, vglobal - ostgthresh) / (rt * ostgtempgamma);
   // the local factor follows the excess of the deposit bin
   if (use_ostltemp and ostltempgamma > 0.0)
      expo -= std::max(0.0, vlocal - vglobal - ostlthresh) / (rt * ostltempgamma);
   return hbias * std::exp(expo);
}

// buildostindex -- rebuild the packed bins and linked-list lookup from the saved
// real centers (eost.f:buildostindex).
void buildOstIndex()
{
   std::fill(osthead.begin(), osthead.end(), 0);
   for (int i = 1; i <= sizelmdahist; ++i)
      ostnext[i] = 0;
   for (int ihist = 1; ihist <= nlmdahist; ++ihist) {
      int il = lmdaBin(lmdalhist[ihist]);
      int jf = flambdaBin(lmdafhist[ihist]);
      int k;
      ijToK(il, jf, nlmda, k);
      osthist[ihist] = k;
      ostnext[ihist] = osthead[gidx(il, jf)];
      osthead[gidx(il, jf)] = ihist;
   }
}

// resizeosthist -- double the history storage, preserving saved data
// (eost.f:resizeosthist). std::vector::resize keeps existing elements.
void resizeOstHist()
{
   int newsize = 2 * sizelmdahist;
   sizelmdahist = newsize;
   osthist.resize(newsize + 1, 0);
   lmdaihist.resize(newsize + 1, 0);
   ostnext.resize(newsize + 1, 0);
   lmdalhist.resize(newsize + 1, 0.0);
   lmdafhist.resize(newsize + 1, 0.0);
   osthhist.resize(newsize + 1, 0.0);
   ostwlhist.resize(newsize + 1, 0.0);
   ostwfhist.resize(newsize + 1, 0.0);
}

// ensureflambda -- grow the flambda grid (and shift/reallocate the four kernel
// grids) if the new dU/dlambda falls too near the current edge (eost.f:ensureflambda).
void ensureFlambda(double dudl)
{
   int iflmda = (int)std::lround(dudl / wflmda) + fli0;

   int nchunk = 100;
   int nbuffer = nchunk;
   int nfcut = (int)(oststdev * maxwfhist / wflmda) + 1 + nbuffer;
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
// numerator/partition-function accumulators, given the thermal energy rt; the
// caller sets the row mean force (eost.f:addkernelpoint).
void addKernelPoint(int ilmda, int iflmda, double e, double ldelta, double fldelta, double sigl2, double sigf2,
   double rt)
{
   int g = gidx(ilmda, iflmda);
   double oldg = gkernel[g];
   double newg = oldg + e;

   // the accumulators carry the largest bias in the row as a common factor,
   // so they are rescaled whenever this point raises it, and the new point
   // then has unit weight
   double vmax = vkernelmax[ilmda];
   bool raised = (newg > vmax);
   if (raised) {
      double scale = std::exp((vmax - newg) / rt);
      lmdafsum[ilmda] *= scale;
      lmdafwt[ilmda] *= scale;
      vmax = newg;
   }
   double oldweight = (oldg == 0.0) ? 0.0 : std::exp((oldg - vmax) / rt);
   double newweight = raised ? 1.0 : std::exp((newg - vmax) / rt);
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
   lmdafsum[ilmda] += flmda * delweight;
   lmdafwt[ilmda] += delweight;
}

// addkernelhist -- spread one saved histogram source over nearby grid bins;
// when dofkern is set it also updates the g kernel derivatives and the f kernel
// accumulators, otherwise only the g kernel and its running maxima
// (eost.f:addkernelhist).
void addKernelHist(int ihist, bool dofkern)
{
   int k = osthist[ihist];
   int lsrc, fsrc;
   kToIj(k, nlmda, lsrc, fsrc);
   double sigl = ostwlhist[ihist];
   double sigf = ostwfhist[ihist];
   double sigl2 = sigl * sigl;
   double sigf2 = sigf * sigf;
   double pref = osthhist[ihist] / (2.0 * dpi * sigl * sigf);
   double sourcefl = lmdafhist[ihist];
   const double rt = units::gasconst * bath::kelvin;

   int nlcut = nbinCut(oststdev * sigl, wlmda);
   int nfcut = nbinCut(oststdev * sigf, wflmda);
   int iflmda1 = std::max(1, fsrc - nfcut);
   int iflmda2 = std::min(nflmda, fsrc + nfcut);

   for (int img = 1; img <= 3; ++img) {
      int llog;
      double sourcel;
      if (img == 1) {
         llog = lsrc;
         sourcel = lmdalhist[ihist];
      } else if (img == 2) {
         llog = 2 - lsrc;
         sourcel = -lmdalhist[ihist];
      } else {
         llog = 2 * nlmda - lsrc;
         sourcel = 2.0 - lmdalhist[ihist];
      }
      int ilmda1 = std::max(1, llog - nlcut);
      int ilmda2 = std::min(nlmda, llog + nlcut);
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
            if (dofkern) {
               addKernelPoint(ilmda, iflmda, e, ldelta, fldelta, sigl2, sigf2, rt);
            } else {
               double newg = gkernel[gidx(ilmda, iflmda)] + e;
               gkernel[gidx(ilmda, iflmda)] = newg;
               vkernelmax[ilmda] = std::max(vkernelmax[ilmda], newg);
            }
         }

         // set the mean force of the row once all its points are added
         if (dofkern)
            lmdafmean[ilmda] = (lmdafwt[ilmda] == 0.0) ? 0.0 : lmdafsum[ilmda] / lmdafwt[ilmda];
      }
   }
}

// buildfkernel -- rebuild the mean force at every lambda bin from the g kernel
// (eost.f:buildfkernel).
void buildFkernel()
{
   const double rt = units::gasconst * bath::kelvin;
   for (int il = 1; il <= nlmda; ++il) {
      double avg = 0;
      double pf = 0;
      // the largest bias in the row is factored out of both sums, where it
      // cancels exactly, to keep the exponential from overflowing
      double vmax = vkernelmax[il];
      for (int jf = 1; jf <= nflmda; ++jf) {
         double g = gkernel[gidx(il, jf)];
         if (g != 0.0) {
            double flmda = (double)(jf - fli0) * wflmda;
            double w = std::exp((g - vmax) / rt);
            avg += flmda * w;
            pf += w;
         }
      }
      lmdafmean[il] = (pf == 0.0) ? 0.0 : avg / pf;
   }
}

// egkernel -- direct linked-list gaussian sum of the g bias and its partials at
// the current (lambda, dedl) (eost.f:egkernel).
void egkernel(double& egbias, double& dgdl, double& dgdfl)
{
   egbias = 0;
   dgdl = 0;
   dgdfl = 0;

   int ilmda = lmdaBin(lambda);
   int iflmda = (int)std::lround(dedl / wflmda) + fli0;
   if (iflmda < 1 || iflmda > nflmda)
      return;

   // use max gaussian widths for conservative bin cutoffs
   int nlcut = nbinCut(oststdev * maxwlhist, wlmda) + 1;
   int nfcut = nbinCut(oststdev * maxwfhist, wflmda) + 1;

   for (int klmda = -nlcut; klmda <= nlcut; ++klmda) {
      int lcenter = ilmda + klmda;
      int lcount, nimg;
      lmdaImgBin(lcenter, lcount, nimg);
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
            double sourcefl = lmdafhist[ihist];
            double fldelta = dedl - sourcefl;
            if (std::fabs(fldelta) <= oststdev * sigf) {
               double sigl2 = sigl * sigl;
               double sigf2 = sigf * sigf;
               double sigl2inv = 1.0 / sigl2;
               double sigf2inv = 1.0 / sigf2;
               double pref = osthhist[ihist] / (2.0 * dpi * sigl * sigf);
               double expfl = 0;
               bool getexpfl = true;
               for (int img = 1; img <= nimg; ++img) {
                  double sourcel = lmdaImgSrc(lcenter, img, lmdalhist[ihist]);
                  double ldelta = lambda - sourcel;
                  if (std::fabs(ldelta) <= oststdev * sigl) {
                     // the flambda factor is found once, when an image first counts
                     if (getexpfl) {
                        double fldelta2 = fldelta * fldelta;
                        expfl = std::exp(-0.5 * fldelta2 * sigf2inv);
                        getexpfl = false;
                     }
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
// derivatives from the grids (eost.f:egkernelinterpolate).
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

// emetabias -- Vbias(lambda) and dVbias/dlambda for the sum of 1D normalized
// metadynamics gaussians, each counted with its two images reflected at the
// lambda end points; only gaussians in the lambda bins within the cutoff of
// oststdev widths are visited, through the bin lookup, and only the first
// nmetahist are counted, so a history replayed gaussian by gaussian gives the
// bias at each stage (eost.f:emetabias).
void eMetaBias(double lmda, double& vbias, double& dvdl)
{
   vbias = 0;
   dvdl = 0;

   if (ostinterpol && nmetahist > 0) {
      eMetaBiasInterpolate(lmda, vbias, dvdl);
      return;
   }

   // use the largest gaussian width for a conservative bin cutoff
   int ilmda = lmdaBin(lmda);
   int nlcut = nbinCut(oststdev * maxwlhist, wlmda) + 1;

   // loop over nearby lambda bins and their mirror images
   for (int klmda = -nlcut; klmda <= nlcut; ++klmda) {
      int lcenter = ilmda + klmda;
      int lcount, nimg;
      lmdaImgBin(lcenter, lcount, nimg);
      if (lcount < 1 || lcount > nlmda)
         continue;

      // add each gaussian of the bin with the image it stands for
      int ihist = metahead[lcount];
      while (ihist != 0) {
         double sig = metawhist[ihist];
         if (ihist <= nmetahist && sig > 0.0) {
            double sig2 = sig * sig;
            double pref = metahhist[ihist] / (sig * std::sqrt(2.0 * dpi));
            for (int img = 1; img <= nimg; ++img) {
               double delta = lmda - lmdaImgSrc(lcenter, img, metalhist[ihist]);
               if (std::fabs(delta) <= oststdev * sig) {
                  double bias = pref * std::exp(-0.5 * delta * delta / sig2);
                  vbias += bias;
                  dvdl -= delta * bias / sig2;
               }
            }
         }
         ihist = metanext[ihist];
      }
   }
}

// addmetagrid -- add one deposited metadynamics gaussian and its two reflected
// images to the bias and slope at the lambda bin centers within the cutoff of
// oststdev widths (eost.f:addmetagrid).
void addMetaGrid(int ihist)
{
   double sig = metawhist[ihist];
   if (sig <= 0.0)
      return;
   if ((int)vmetagrid.size() < nlmda + 1 || (int)dvmetagrid.size() < nlmda + 1)
      return;
   double sig2 = sig * sig;
   double pref = metahhist[ihist] / (sig * std::sqrt(2.0 * dpi));
   double cut = oststdev * sig;
   double src[3] = {metalhist[ihist], -metalhist[ihist], 2.0 - metalhist[ihist]};

   // spread the gaussian and its images over the bin centers within the cutoff
   for (int img = 0; img < 3; ++img) {
      double x1 = (src[img] - cut) / wlmda + 1.0;
      double x2 = (src[img] + cut) / wlmda + 1.0;
      if (x1 > (double)nlmda || x2 < 1.0)
         continue;
      int il1 = (int)std::max(1.0, x1);
      int il2 = std::min(nlmda, (int)std::min((double)nlmda, x2) + 1);
      for (int il = il1; il <= il2; ++il) {
         double lmda = (double)(il - 1) * wlmda;
         double delta = lmda - src[img];
         if (std::fabs(delta) <= cut) {
            double bias = pref * std::exp(-0.5 * delta * delta / sig2);
            vmetagrid[il] += bias;
            dvmetagrid[il] -= delta * bias / sig2;
         }
      }
   }
}

// buildmetaindex -- rebuild the linked-list lookup of the saved metadynamics
// gaussians by lambda bin, in the order deposited (eost.f:buildmetaindex).
void buildMetaIndex()
{
   std::fill(metahead.begin(), metahead.end(), 0);
   std::fill(metanext.begin(), metanext.end(), 0);
   for (int ihist = 1; ihist <= nmetahist; ++ihist) {
      int ilmda = lmdaBin(metalhist[ihist]);
      metanext[ihist] = metahead[ilmda];
      metahead[ilmda] = ihist;
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

// metadeltag -- G(1) - G(0) = -Vbias(1) + Vbias(0), with the end point bias read
// from the grid, which holds the exact bias at the bin centers for every
// gaussian added so far (eost.f:metadeltag).
double metaDeltaG()
{
   if ((int)vmetagrid.size() < nlmda + 1 || nlmda < 1)
      return 0.0;
   return -vmetagrid[nlmda] + vmetagrid[1];
}

void resizeMeta()
{
   int newsize = 2 * sizemetahist;
   sizemetahist = newsize;
   metalhist.resize(newsize + 1, 0.0);
   metahhist.resize(newsize + 1, 0.0);
   metawhist.resize(newsize + 1, 0.0);
   metaihist.resize(newsize + 1, 0);
   metanext.resize(newsize + 1, 0);
}

// nbincut -- smallest number of bins of the given width that together span the
// given cutoff distance (eost.f:nbincut).
int nbinCut(double cut, double width)
{
   int n = (int)(cut / width);
   if ((double)n * width < cut)
      n = n + 1;
   return n;
}

// lmdaimgbin -- map a lambda bin index lcenter, which may lie beyond either end
// of the lambda range, onto the bin lcount whose gaussians have their mirror
// image in it; nimg is two in each end bin, for the gaussian and its
// reflection, otherwise one (eost.f:lmdaimgbin).
void lmdaImgBin(int lcenter, int& lcount, int& nimg)
{
   lcount = lcenter;
   if (lcount < 1)
      lcount = 2 - lcount;
   else if (lcount > nlmda)
      lcount = 2 * nlmda - lcount;
   nimg = 1;
   if (lcenter == 1 || lcenter == nlmda)
      nimg = 2;
}

// lmdaimgsrc -- lambda center of image img of a gaussian centered at lsrc, as
// counted in lambda bin lcenter by lmdaImgBin (eost.f:lmdaimgsrc).
double lmdaImgSrc(int lcenter, int img, double lsrc)
{
   if (lcenter < 1)
      return -lsrc;
   else if (lcenter > nlmda)
      return 2.0 - lsrc;
   else if (img == 2 && lcenter == 1)
      return -lsrc;
   else if (img == 2 && lcenter == nlmda)
      return 2.0 - lsrc;
   else
      return lsrc;
}

// Device scaffold (disabled). A future GPU port of the bicubic-Hermite bias
// evaluator would be dispatched here, following the pattern in ost.cpp.
// See src/acc/eost.cpp and src/cu/eost.cu. Enable together with those files.
//   TINKER_FVOID2(acc0, cu1, eostBiasGrid, /* grid ptrs, sample, outputs */);

// buildgkernel -- clear stale values and rebuild the g kernel from scratch over
// all saved sources (eost.f:buildgkernel).
void buildGkernel()
{
   std::fill(gkernel.begin(), gkernel.end(), 0.0);
   std::fill(vkernelmax.begin(), vkernelmax.end(), 0.0);
   for (int ihist = 1; ihist <= nlmdahist; ++ihist)
      addKernelHist(ihist, false);
}

// buildkernels -- clear and rebuild both kernels and the free-energy
// accumulators from scratch over all saved sources (eost.f:buildkernels).
void buildKernels()
{
   std::fill(gkernel.begin(), gkernel.end(), 0.0);
   std::fill(gfkernel.begin(), gfkernel.end(), 0.0);
   std::fill(glfkernel.begin(), glfkernel.end(), 0.0);
   std::fill(glkernel.begin(), glkernel.end(), 0.0);
   std::fill(lmdafmean.begin(), lmdafmean.end(), 0.0);
   std::fill(lmdafsum.begin(), lmdafsum.end(), 0.0);
   std::fill(lmdafwt.begin(), lmdafwt.end(), 0.0);
   std::fill(vkernelmax.begin(), vkernelmax.end(), 0.0);
   for (int ihist = 1; ihist <= nlmdahist; ++ihist)
      addKernelHist(ihist, true);
}

// updategkernel / updatekernels -- spread only the most recently saved source
// into the current grids (eost.f:updategkernel / updatekernels).
void updateGkernel()
{
   if (nlmdahist > 0)
      addKernelHist(nlmdahist, false);
}

void updateKernels()
{
   if (nlmdahist > 0)
      addKernelHist(nlmdahist, true);
}

void eostData(RcOp op)
{
   if (!(use_ost || use_meta))
      return;

   if (op & RcOp::DEALLOC) {
      osthist.clear();
      lmdaihist.clear();
      ostnext.clear();
      osthead.clear();
      lmdalhist.clear();
      lmdafhist.clear();
      osthhist.clear();
      ostwlhist.clear();
      ostwfhist.clear();
      lmdallist.clear();
      lmdaflist.clear();
      gkernel.clear();
      glkernel.clear();
      gfkernel.clear();
      glfkernel.clear();
      lmdafmean.clear();
      lmdafsum.clear();
      lmdafwt.clear();
      vkernelmax.clear();
      metalhist.clear();
      metahhist.clear();
      metawhist.clear();
      metaihist.clear();
      metanext.clear();
      metahead.clear();
      vmetagrid.clear();
      dvmetagrid.clear();
   }

   if (op & RcOp::INIT) {
      bgbias = 0;
      bdgdl = 0;
      bdgdfl = 0;
      bostlmda = 0;
      lmdadfdl = 0;

      // Mirror the Fortran mutate allocation/initialization (mutate.f:mutate_ost, mutate_meta).
      if (use_ost) {
         sizelmdahist = 10000;
         nlmdahist = 0;
         osthist.assign(sizelmdahist + 1, 0);
         lmdaihist.assign(sizelmdahist + 1, 0);
         ostnext.assign(sizelmdahist + 1, 0);
         osthead.assign((size_t)nlmda * nflmda, 0);
         lmdallist.assign(lmdaintv, 0.0);
         lmdaflist.assign(lmdaintv, 0.0);
         lmdalhist.assign(sizelmdahist + 1, 0.0);
         lmdafhist.assign(sizelmdahist + 1, 0.0);
         osthhist.assign(sizelmdahist + 1, 0.0);
         ostwlhist.assign(sizelmdahist + 1, 0.0);
         ostwfhist.assign(sizelmdahist + 1, 0.0);
         lmdafmean.assign(nlmda + 1, 0.0);
         lmdafsum.assign(nlmda + 1, 0.0);
         lmdafwt.assign(nlmda + 1, 0.0);
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
         metanext.assign(sizemetahist + 1, 0);
         metahead.assign(nlmda + 1, 0);
         lmdallist.assign(lmdaintv, 0.0);
         vmetagrid.assign(nlmda + 1, 0.0);
         dvmetagrid.assign(nlmda + 1, 0.0);
      }
   }
}

void eostBias(int vers)
{
   // dedl is the unbiased dU/dlambda for this configuration, accumulated by the
   // energy terms, each already in main lambda units. Nothing below writes
   // it, so it stays valid until elmdaDyn consumes it after energy() returns;
   // zeroEGV zeroes it again at the top of the next evaluation.
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
   efreeLmda(lambda, bostlmda, lmdadfdl);

   if (vers & calc::energy)
      esum += bgbias - bostlmda;

   // The g bias depends on dU/dlambda, whose gradient/virial are dfdl* and
   // dvirdl, so these are the bias's second-order Cartesian terms. The force term
   // runs on device via sumGradient.
   const int dvers = lmdaDerivVers(vers, use_dlmda);
   if (dvers & calc::grad_dlmda)
      sumGradient(bdgdfl, gx, gy, gz, dfdlx, dfdly, dfdlz);
   if ((dvers & calc::virial_dlmda) and (vers & calc::virial))
      for (int k = 0; k < 9; ++k)
         vir[k] += bdgdfl * dvirdl[k];
}

// ostDeposit -- records one accepted interval of orthogonal space tempering as
// a biasing gaussian centered on the interval lambda and dU/dlambda averages,
// then updates the bias kernels and the free energy estimate
// (eost.f:ostdeposit).
void ostDeposit(int istep)
{
   // ensure histogram contains the unbiased dU/dlambda value
   int ilmda = lmdaBin(lmdaavg);
   maxwlhist = std::max(maxwlhist, wlhist);
   maxwfhist = std::max(maxwfhist, wfhist);
   ensureFlambda(dedlavg);
   int iflmda = flambdaBin(dedlavg);

   // ensure histogram array is sufficiently large
   nlmdahist = nlmdahist + 1;
   if (nlmdahist > sizelmdahist)
      resizeOstHist();
   int k;
   ijToK(ilmda, iflmda, nlmda, k);

   // save histogram information
   osthist[nlmdahist] = k;
   lmdaihist[nlmdahist] = istep;
   lmdalhist[nlmdahist] = lmdaavg;
   lmdafhist[nlmdahist] = dedlavg;
   osthhist[nlmdahist] = temperedHeight(ostVminimax(), vkernelmax[ilmda]);
   ostwlhist[nlmdahist] = wlhist;
   ostwfhist[nlmdahist] = wfhist;
   ostnext[nlmdahist] = osthead[gidx(ilmda, iflmda)];
   osthead[gidx(ilmda, iflmda)] = nlmdahist;

   if (true) {
      double vmm = ostVminimax();
      double th = temperedHeight(vmm, vkernelmax[ilmda]);
      printf("istep: %i\n", istep);
      printf("ostlmda  avg, std: %8.4f %8.4e\n", lmdaavg, lmdastd);
      printf("ostdedl  avg, std: %8.4f %8.4e\n", dedlavg, dedlstd);
      printf("vminmax temperedHeight: %8.4e %8.4e\n", vmm, th);
      printf("\n");
   }

   if (fastkernel) {
      updateKernels();
   } else {
      updateGkernel();
      buildFkernel();
   }
   lmdadeltag = efreeTot();
}

// metaDeposit -- records one interval of lambda metadynamics as a gaussian
// centered on the interval lambda average, then updates the bin lookup, the
// bias grid and the free energy estimate; only the global tempering factor is
// applied, since the free energy is read from the bias itself, and a local
// factor would keep the bias from filling barriers above its threshold
// (eost.f:metadeposit).
void metaDeposit(int istep)
{
   // save the new gaussian with the global tempering factor alone
   nmetahist = nmetahist + 1;
   if (nmetahist > sizemetahist)
      resizeMeta();
   double vglobal = metaVminimax();
   metalhist[nmetahist] = lmdaavg;
   metahhist[nmetahist] = temperedHeight(vglobal, vglobal);
   metawhist[nmetahist] = wlhist;
   metaihist[nmetahist] = istep;
   maxwlhist = std::max(maxwlhist, wlhist);

   // link the gaussian into its lambda bin, then add it to the grid
   int ilmda = lmdaBin(lmdaavg);
   metanext[nmetahist] = metahead[ilmda];
   metahead[ilmda] = nmetahist;
   addMetaGrid(nmetahist);
   lmdadeltag = metaDeltaG();
}
}
