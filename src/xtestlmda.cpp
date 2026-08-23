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
#include <algorithm>
#include <cmath>
#include <string>
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

struct LambdaEvaluation
{
   LambdaEnergy energy;
   std::vector<double> gradient;
   double virial[9] = {};
};

enum class LambdaVariable
{
   MAIN,
   VDW,
   ELE,
   POL
};

struct FdLambdaState
{
   bool useMain;
   double main;
   double vdw;
   double ele;
   double pol;
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

static FdLambdaState captureFdLambdaState()
{
   bool useMain = use_mainlmda;
   return {useMain, useMain ? lambda : 0.0, vlam, elam, plam};
}

struct ScopedDlmdaOff
{
   bool saved, saved_e, saved_p, saved_v;
   ScopedDlmdaOff()
      : saved(use_dlmda)
      , saved_e(use_edlmda)
      , saved_p(use_pdlmda)
      , saved_v(use_vdlmda)
   {
      use_dlmda = false;
      use_edlmda = false;
      use_pdlmda = false;
      use_vdlmda = false;
   }
   ~ScopedDlmdaOff()
   {
      use_dlmda = saved;
      use_edlmda = saved_e;
      use_pdlmda = saved_p;
      use_vdlmda = saved_v;
   }
};

// Applies a displacement from the captured point. A main lambda is mapped to
// all sub-lambdas by energy(); without one, exactly one fixed sub-lambda moves.
static void setFdOffset(const FdLambdaState& state, LambdaVariable variable, double delta)
{
   if (state.useMain) {
      lambda = state.main + delta;
      return;
   }

   double vdw = state.vdw;
   double ele = state.ele;
   double pol = state.pol;
   if (variable == LambdaVariable::VDW)
      vdw += delta;
   else if (variable == LambdaVariable::ELE)
      ele += delta;
   else if (variable == LambdaVariable::POL)
      pol += delta;
   setLambda(vdw, ele, pol);
}

static void restoreFdLambdaState(const FdLambdaState& state)
{
   if (state.useMain) {
      lambda = state.main;
      mapSubLambda();
   } else {
      setLambda(state.vdw, state.ele, state.pol);
   }
}

static LambdaEvaluation evaluateAtOffset(const FdLambdaState& state, LambdaVariable variable, double delta)
{
   setFdOffset(state, variable, delta);
   energy(calc::v1);

   LambdaEvaluation eout;
   copyEnergy(calc::energy, &eout.energy.total);
   eout.energy.vdw = energy_vdw;
   eout.energy.mpole = energy_em;
   eout.energy.polar = energy_ep;
   copyGradientFlat(calc::grad, eout.gradient);
   for (int k = 0; k < 9; ++k)
      eout.virial[k] = vir[k];
   return eout;
}

static double lambdaEnergy(const LambdaEvaluation& e, LambdaVariable variable)
{
   if (variable == LambdaVariable::VDW)
      return e.energy.vdw;
   if (variable == LambdaVariable::ELE)
      return e.energy.mpole;
   if (variable == LambdaVariable::POL)
      return e.energy.polar;
   return e.energy.total;
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
   const bool keylmda = use_dlmda;
   const bool astpolar = use_epast;
   r.dfdl.assign(3 * n, 0.0);
   r.ndfdl.assign(3 * n, 0.0);

   // ---- Analytical lambda derivatives --------------------------------------
   if (opts.analyt and keylmda) {
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

      if (lmdaDerivMask(calc::v1, keylmda) & calc::grad_dlmda)
         copyGradientFlat(calc::grad, r.dfdl, dfdlx, dfdly, dfdlz);
   }

   // ---- Numerical lambda derivatives ---------------------------------------
   if (opts.numer and keylmda) {
      const FdLambdaState state = captureFdLambdaState();
      ScopedDlmdaOff dlmda_off;
      const double eps = opts.eps;
      const double eps2 = eps * eps;
      LambdaEvaluation center = evaluateAtOffset(state, LambdaVariable::MAIN, 0.0);

      if (state.useMain) {
         const bool atLower = state.main - eps < 0;

         LambdaEvaluation plus = evaluateAtOffset(state, LambdaVariable::MAIN, eps);
         LambdaEvaluation minus, plus2;
         if (atLower)
            plus2 = evaluateAtOffset(state, LambdaVariable::MAIN, 2.0 * eps);
         else
            minus = evaluateAtOffset(state, LambdaVariable::MAIN, -eps);

         const LambdaVariable variables[] = {
            LambdaVariable::MAIN, LambdaVariable::VDW, LambdaVariable::ELE, LambdaVariable::POL};
         for (int k = 0; k < 4; ++k) {
            double ep = lambdaEnergy(plus, variables[k]);
            double ec = lambdaEnergy(center, variables[k]);
            if (atLower) {
               double ep2 = lambdaEnergy(plus2, variables[k]);
               r.ndedl[k] = (-3.0 * ec + 4.0 * ep - ep2) / (2.0 * eps);
               r.nd2edl2[k] = (ec - 2.0 * ep + ep2) / eps2;
            } else {
               double em = lambdaEnergy(minus, variables[k]);
               r.ndedl[k] = (ep - em) / (2.0 * eps);
               r.nd2edl2[k] = (ep - 2.0 * ec + em) / eps2;
            }
         }
         if (atLower) {
            for (int k = 0; k < 3 * n; ++k)
               r.ndfdl[k] = (-3.0 * center.gradient[k] + 4.0 * plus.gradient[k] - plus2.gradient[k]) / (2.0 * eps);
            for (int k = 0; k < 9; ++k)
               r.ndvirdl[k] = (-3.0 * center.virial[k] + 4.0 * plus.virial[k] - plus2.virial[k]) / (2.0 * eps);
         } else {
            for (int k = 0; k < 3 * n; ++k)
               r.ndfdl[k] = (plus.gradient[k] - minus.gradient[k]) / (2.0 * eps);
            for (int k = 0; k < 9; ++k)
               r.ndvirdl[k] = (plus.virial[k] - minus.virial[k]) / (2.0 * eps);
         }
      } else {
         // Fixed sub-lambdas are independent coordinates. Test each derivative
         // at its own configured value, then sum them for the total derivative.
         const LambdaVariable variables[] = {LambdaVariable::VDW, LambdaVariable::ELE, LambdaVariable::POL};
         for (int j = 0; j < 3; ++j) {
            LambdaVariable variable = variables[j];
            LambdaEvaluation plus = evaluateAtOffset(state, variable, eps);
            LambdaEvaluation minus = evaluateAtOffset(state, variable, -eps);
            double ep = lambdaEnergy(plus, variable);
            double ec = lambdaEnergy(center, variable);
            double em = lambdaEnergy(minus, variable);
            r.ndedl[j + 1] = (ep - em) / (2.0 * eps);
            r.nd2edl2[j + 1] = (ep - 2.0 * ec + em) / eps2;
            for (int k = 0; k < 3 * n; ++k)
               r.ndfdl[k] += (plus.gradient[k] - minus.gradient[k]) / (2.0 * eps);
            for (int k = 0; k < 9; ++k)
               r.ndvirdl[k] += (plus.virial[k] - minus.virial[k]) / (2.0 * eps);
         }
         for (int k = 1; k < 4; ++k) {
            r.ndedl[0] += r.ndedl[k];
            r.nd2edl2[0] += r.nd2edl2[k];
         }
      }

      if (astpolar) {
         for (int k = 0; k < 4; ++k)
            r.nd2edl2[k] = 0.0;
         std::fill(r.ndfdl.begin(), r.ndfdl.end(), 0.0);
         for (int k = 0; k < 9; ++k)
            r.ndvirdl[k] = 0.0;
      }

      restoreFdLambdaState(state);
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
