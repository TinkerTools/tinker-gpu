#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/rwcrd.h"
#include "ff/thermint.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/ioread.h"
#include "tool/xtesthelper.h"
#include <cmath>
#include <string>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/usage.hh>
#include <tinker/routines.h>
#include <vector>

#include "tinker9.h"

namespace tinker {
static constexpr double default_lambda_eps = 1.0e-2;

static FdTestOptions readOptions()
{
   return readFdTestOptions("\n"
                            " Compute the Analytical Derivative [Y] :  ",
      "\n"
      " Compute the Numerical Derivative [Y] :   ",
      default_lambda_eps, "Lambda");
}

struct LambdaEnergy
{
   energy_prec total;
   energy_prec vdw;
   energy_prec mpole;
   energy_prec polar;
};

static void setLambda(double vlambda, double elambda, double plambda)
{
   mutant::vlambda = vlambda;
   vlam = vlambda;
   mutant::elambda = elambda;
   elam = elambda;
   mutant::plambda = plambda;
   plam = plambda;
}

// The lambda the finite differences perturb. When OST, metadynamics, or TI owns
// the main lambda, energy() re-derives vlam/elam/plam from it through
// mapSubLambda(), so writing the sub-lambdas directly would have no effect.
// With no owner there is no main lambda, and the sub-lambdas move together.
static double fdLambda()
{
   return useLmdaChain() ? mainLambda() : elam;
}

static void setFdLambda(double lambda)
{
   if (use_ost or use_meta)
      ostlambda = lambda;
   else if (use_ti)
      tilmda = lambda;
   else
      setLambda(lambda, lambda, lambda);
}

// Evaluate the potential energy at a given main lambda value.
static LambdaEnergy energyAtLambda(double lambda)
{
   setFdLambda(lambda);
   energy(calc::energy);
   LambdaEnergy eout;
   copyEnergy(calc::energy, &eout.total);
   eout.vdw = energy_vdw;
   eout.mpole = energy_em;
   eout.polar = energy_ep;
   return eout;
}

static void printDerivRow(FILE* out, const char* title, const char* l0, const char* l1, const char* l2, const char* l3,
   double v0, double v1, double v2, double v3)
{
   print(out, "\n%-36s%14s%14s%14s%14s\n", title, l0, l1, l2, l3);
   print(out, "%36s%14.6f%14.6f%14.6f%14.6f\n", "", v0, v1, v2, v3);
}

int testlmdaFlags(const FdTestOptions& opts)
{
   int flags = calc::xyz + calc::mass + calc::energy + calc::analyz;
   if (opts.analyt or opts.numer)
      flags += calc::grad + calc::virial;
   return flags;
}

TestlmdaResult testlmdaEvaluate(const FdTestOptions& opts)
{
   TestlmdaResult r;

   // ---- Analytical lambda derivatives --------------------------------------
   if (opts.analyt) {
      energy(calc::v1);
      r.dedl[0] = dedl;
      r.dedl[1] = devdl;
      r.dedl[2] = demdl;
      r.dedl[3] = depdl;
      r.d2edl2[0] = d2edl2;
      r.d2edl2[1] = d2evdl2;
      r.d2edl2[2] = d2emdl2;
      r.d2edl2[3] = d2epdl2;
      for (int k = 0; k < 9; ++k)
         r.dvirdl[k] = dvirdl[k];

      copyGradientFlat(calc::grad, r.dfdl, dfsumdlx, dfsumdly, dfsumdlz);
   }

   // ---- Numerical lambda derivatives ---------------------------------------
   if (opts.numer) {
      const double lam0 = fdLambda();
      const double eps = opts.eps;

      // Scalar first/second derivatives from three energy evaluations.
      LambdaEnergy e2 = energyAtLambda(lam0 + eps);
      LambdaEnergy e0 = energyAtLambda(lam0 - eps);
      LambdaEnergy e1 = energyAtLambda(lam0);
      r.ndedl[0] = (e2.total - e0.total) / (2.0 * eps);
      r.ndedl[1] = (e2.vdw - e0.vdw) / (2.0 * eps);
      r.ndedl[2] = (e2.mpole - e0.mpole) / (2.0 * eps);
      r.ndedl[3] = (e2.polar - e0.polar) / (2.0 * eps);
      r.nd2edl2[0] = (e2.total - 2.0 * e1.total + e0.total) / (eps * eps);
      r.nd2edl2[1] = (e2.vdw - 2.0 * e1.vdw + e0.vdw) / (eps * eps);
      r.nd2edl2[2] = (e2.mpole - 2.0 * e1.mpole + e0.mpole) / (eps * eps);
      r.nd2edl2[3] = (e2.polar - 2.0 * e1.polar + e0.polar) / (eps * eps);

      // Per-atom force derivative and virial derivative via central
      // difference of the gradient/virial at lambda +/- eps.
      std::vector<double> gpx(n), gpy(n), gpz(n), gmx(n), gmy(n), gmz(n);

      setFdLambda(lam0 + eps);
      energy(calc::v1);
      copyGradient(calc::grad, gpx.data(), gpy.data(), gpz.data());
      double vplus[9];
      for (int k = 0; k < 9; ++k)
         vplus[k] = vir[k];

      setFdLambda(lam0 - eps);
      energy(calc::v1);
      copyGradient(calc::grad, gmx.data(), gmy.data(), gmz.data());
      for (int k = 0; k < 9; ++k)
         r.ndvirdl[k] = (vplus[k] - vir[k]) / (2.0 * eps);

      r.ndfdl.resize(3 * n);
      for (int i = 0; i < n; ++i) {
         r.ndfdl[3 * i + 0] = (gpx[i] - gmx[i]) / (2.0 * eps);
         r.ndfdl[3 * i + 1] = (gpy[i] - gmy[i]) / (2.0 * eps);
         r.ndfdl[3 * i + 2] = (gpz[i] - gmz[i]) / (2.0 * eps);
      }

      // Restore the original lambda value.
      setFdLambda(lam0);
   }

   return r;
}

void testlmdaPrint(FILE* out, const FdTestOptions& opts, const TestlmdaResult& r, int digits)
{
   auto fmt = gradientPrintFormat(digits, "dFx/dL", "dFy/dL", "dFz/dL");

   if (opts.analyt)
      printDerivRow(out, " Analytical Lambda Derivatives :", "dE/dL", "dEV/dL", "dEM/dL", "dEP/dL", r.dedl[0],
         r.dedl[1], r.dedl[2], r.dedl[3]);
   if (opts.numer)
      printDerivRow(out, " Numerical Lambda Derivatives : ", "dE/dL", "dEV/dL", "dEM/dL", "dEP/dL", r.ndedl[0],
         r.ndedl[1], r.ndedl[2], r.ndedl[3]);

   if (opts.analyt)
      printDerivRow(out, " Analytical 2nd Lambda Derivatives :", "d2E/dL2", "d2EV/dL2", "d2EM/dL2", "d2EP/dL2",
         r.d2edl2[0], r.d2edl2[1], r.d2edl2[2], r.d2edl2[3]);
   if (opts.numer)
      printDerivRow(out, " Numerical 2nd Lambda Derivatives : ", "d2E/dL2", "d2EV/dL2", "d2EM/dL2", "d2EP/dL2",
         r.nd2edl2[0], r.nd2edl2[1], r.nd2edl2[2], r.nd2edl2[3]);

   // Per-atom lambda gradient breakdown. Fortran testlmda normalizes the RMS rows
   // by the number of active atoms rather than by all of them.
   double rdenom = std::sqrt((double)(usage::nuse > 0 ? usage::nuse : 1));
   printGradientTable(out, "Lambda Gradient Breakdown over Individual Atoms", fmt, opts, r.dfdl, r.ndfdl, digits,
      rdenom);

   // Virial derivative dV/dL.
   if (opts.analyt)
      // Fortran testlmda fmt 230/240 indents the continuation rows by 27.
      printMatrix(out, "Analytical dV/dL", 8, r.dvirdl, 27);
   if (opts.numer)
      printMatrix(out, "Numerical dV/dL", 9, r.ndvirdl, 27);
}

void xTestlmda(int, char**)
{
   initial();
   int ixyz;
   tinker_f_getcart(&ixyz);
   tinker_f_mechanic();
   dlmda::use_dlmda = 1;
   mechanic2();

   auto out = stdout;

   FdTestOptions opts = readOptions();

   rc_flag = testlmdaFlags(opts);
   initialize();

   int digits = inform::digits;

   FstrView fsw = files::filename;
   std::string fname = fsw.trim();
   int nframe_processed = 0;
   int done = 0;
   auto ipt = CrdReader(fname);
   do {
      done = ipt.readCurrent();
      nblistRefresh();
      nframe_processed++;
      if (nframe_processed > 1)
         print(out, "\n Analysis for Archive Structure :%16d\n", nframe_processed);

      testlmdaPrint(out, opts, testlmdaEvaluate(opts), digits);
   } while (not done);

   finish();
   tinker_f_final();
}
}
