#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/nblist.h"
#include "ff/rwcrd.h"
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
   energy_prec elec;
};

static void setLambda(double vlambda, double elambda)
{
   mutant::vlambda = vlambda;
   vlam = vlambda;
   mutant::elambda = elambda;
}

// Evaluate the potential energy at a given lambda value
static LambdaEnergy energyAtLambda(double vlambda, double elambda)
{
   setLambda(vlambda, elambda);
   energy(calc::energy);
   LambdaEnergy eout;
   copyEnergy(calc::energy, &eout.total);
   eout.vdw = energy_vdw;
   eout.elec = energy_elec;
   return eout;
}

static void printDerivRow(FILE* out, const char* title, const char* l0, const char* l1, const char* l2, const char* l3,
   double v0, double v1, double v2, double v3)
{
   print(out, "\n%-36s%14s%14s%14s%14s\n", title, l0, l1, l2, l3);
   print(out, "%36s%14.6f%14.6f%14.6f%14.6f\n", "", v0, v1, v2, v3);
}

void xTestlmda(int, char**)
{
   initial();
   int ixyz;
   tinker_f_getcart(&ixyz);
   tinker_f_mechanic();
   mechanic2();

   auto out = stdout;

   dlmda::use_dlmda = 1;

   FdTestOptions opts = readOptions();

   int flags = calc::xyz + calc::mass + calc::energy + calc::analyz;
   if (opts.analyt or opts.numer)
      flags += calc::grad + calc::virial;

   rc_flag = flags;
   initialize();

   int digits = inform::digits;
   auto fmt = gradientPrintFormat(digits, "dFx/dL", "dFy/dL", "dFz/dL");

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

      // ---- Analytical lambda derivatives -----------------------------------
      double adedl = 0, adevdl = 0, ademdl = 0, adepdl = 0;
      double ad2edl2 = 0, ad2evdl2 = 0, ad2emdl2 = 0, ad2epdl2 = 0;
      double advirdl[9] = {0};
      std::vector<double> adfx, adfy, adfz;
      if (opts.analyt) {
         energy(calc::v1);
         adedl = dedl;
         adevdl = devdl;
         ademdl = demdl;
         ad2edl2 = d2edl2;
         ad2evdl2 = d2evdl2;
         ad2emdl2 = d2emdl2;
         for (int k = 0; k < 9; ++k)
            advirdl[k] = dvirdl[k];
         adfx.resize(n);
         adfy.resize(n);
         adfz.resize(n);
         copyGradient(calc::grad, adfx.data(), adfy.data(), adfz.data(), dfsumdlx, dfsumdly, dfsumdlz);
      }

      // ---- Numerical lambda derivatives ------------------------------------
      double ndedl = 0, ndevdl = 0, ndemdl = 0, ndepdl = 0;
      double nd2edl2 = 0, nd2evdl2 = 0, nd2emdl2 = 0, nd2epdl2 = 0;
      double ndvirdl[9] = {0};
      std::vector<double> ndfx, ndfy, ndfz;
      if (opts.numer) {
         double el0 = mutant::elambda;
         double vl0 = mutant::vlambda;
         double eps = opts.eps;

         // Scalar first/second derivatives from three energy evaluations.
         LambdaEnergy e2 = energyAtLambda(vl0 + eps, el0 + eps);
         LambdaEnergy e0 = energyAtLambda(vl0 - eps, el0 - eps);
         LambdaEnergy e1 = energyAtLambda(vl0, el0);
         ndedl = (e2.total - e0.total) / (2.0 * eps);
         ndevdl = (e2.vdw - e0.vdw) / (2.0 * eps);
         ndemdl = (e2.elec - e0.elec) / (2.0 * eps);
         nd2edl2 = (e2.total - 2.0 * e1.total + e0.total) / (eps * eps);
         nd2evdl2 = (e2.vdw - 2.0 * e1.vdw + e0.vdw) / (eps * eps);
         nd2emdl2 = (e2.elec - 2.0 * e1.elec + e0.elec) / (eps * eps);

         // Per-atom force derivative and virial derivative via central
         // difference of the gradient/virial at lambda +/- eps.
         std::vector<double> gpx(n), gpy(n), gpz(n), gmx(n), gmy(n), gmz(n);

         setLambda(vl0 + eps, el0 + eps);
         energy(calc::v1);
         copyGradient(calc::grad, gpx.data(), gpy.data(), gpz.data());
         double vplus[9];
         for (int k = 0; k < 9; ++k)
            vplus[k] = vir[k];

         setLambda(vl0 - eps, el0 - eps);
         energy(calc::v1);
         copyGradient(calc::grad, gmx.data(), gmy.data(), gmz.data());
         for (int k = 0; k < 9; ++k)
            ndvirdl[k] = (vplus[k] - vir[k]) / (2.0 * eps);

         ndfx.resize(n);
         ndfy.resize(n);
         ndfz.resize(n);
         for (int i = 0; i < n; ++i) {
            ndfx[i] = (gpx[i] - gmx[i]) / (2.0 * eps);
            ndfy[i] = (gpy[i] - gmy[i]) / (2.0 * eps);
            ndfz[i] = (gpz[i] - gmz[i]) / (2.0 * eps);
         }

         // Restore the original lambda value.
         setLambda(vl0, el0);
      }

      // ---- Output ----------------------------------------------------------
      if (opts.analyt)
         printDerivRow(out, " Analytical Lambda Derivatives :", "dE/dL", "dEV/dL", "dEM/dL", "dEP/dL", adedl, adevdl,
            ademdl, adepdl);
      if (opts.numer)
         printDerivRow(out, " Numerical Lambda Derivatives : ", "dE/dL", "dEV/dL", "dEM/dL", "dEP/dL", ndedl, ndevdl,
            ndemdl, ndepdl);

      if (opts.analyt)
         printDerivRow(out, " Analytical 2nd Lambda Derivatives :", "d2E/dL2", "d2EV/dL2", "d2EM/dL2", "d2EP/dL2",
            ad2edl2, ad2evdl2, ad2emdl2, ad2epdl2);
      if (opts.numer)
         printDerivRow(out, " Numerical 2nd Lambda Derivatives : ", "d2E/dL2", "d2EV/dL2", "d2EM/dL2", "d2EP/dL2",
            nd2edl2, nd2evdl2, nd2emdl2, nd2epdl2);

      // Per-atom lambda gradient breakdown.
      if (opts.analyt or opts.numer) {
         print(out, "\n Lambda Gradient Breakdown over Individual Atoms :\n");
         print(out, fmt.header, "");
      }
      double totnorm = 0, ntotnorm = 0;
      for (int i = 0; i < n; ++i) {
         if (opts.analyt) {
            totnorm += adfx[i] * adfx[i] + adfy[i] * adfy[i] + adfz[i] * adfz[i];
            printGradientRow(out, fmt.row, "Anlyt", i + 1, adfx[i], adfy[i], adfz[i]);
         }
         if (opts.numer) {
            ntotnorm += ndfx[i] * ndfx[i] + ndfy[i] * ndfy[i] + ndfz[i] * ndfz[i];
            printGradientRow(out, fmt.row, "Numer", i + 1, ndfx[i], ndfy[i], ndfz[i]);
         }
      }

      if (opts.analyt or opts.numer) {
         print(out, "\n\n Total Gradient Norm and RMS Gradient per Atom :\n");
         const char* fmt_summary = "\n %1$s      %2$-30s%3$*4$.*5$f";
         const int len3 = 13 + digits;
         // Fortran testlmda normalizes by the number of active atoms.
         double rdenom = std::sqrt((double)(usage::nuse > 0 ? usage::nuse : 1));

         totnorm = std::sqrt(totnorm);
         ntotnorm = std::sqrt(ntotnorm);
         if (opts.analyt)
            printSummaryRow(out, fmt_summary, "Anlyt", "Total Gradient Norm Value", totnorm, len3, digits);
         if (opts.numer)
            printSummaryRow(out, fmt_summary, "Numer", "Total Gradient Norm Value", ntotnorm, len3, digits);
         print(out, "\n");

         if (opts.analyt)
            printSummaryRow(out, fmt_summary, "Anlyt", "RMS Gradient over All Atoms", totnorm / rdenom, len3, digits);
         if (opts.numer)
            printSummaryRow(out, fmt_summary, "Numer", "RMS Gradient over All Atoms", ntotnorm / rdenom, len3, digits);
         print(out, "\n");
      }

      // Virial derivative dV/dL.
      if (opts.analyt)
         // Fortran testlmda fmt 230/240 indents the continuation rows by 27.
         printMatrix(out, "Analytical dV/dL", 8, advirdl, 27);
      if (opts.numer)
         printMatrix(out, "Numerical dV/dL", 9, ndvirdl, 27);
   } while (not done);

   finish();
   tinker_f_final();
}
}
