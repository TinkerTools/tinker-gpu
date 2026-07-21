#include "ff/atom.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/rwcrd.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/ioread.h"
#include "tool/xtesthelper.h"
#include <array>
#include <cmath>
#include <string>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/solpot.hh>
#include <tinker/routines.h>
#include <vector>

#include "tinker9.h"

namespace tinker {
static energy_prec numericalEnergy()
{
   syncXyzFromHost();
   return evaluateEnergy();
}

static double defaultFiniteDifferenceStep()
{
   double eps = 0.01;
   std::string solvtyp = FstrView(solpot::solvtyp).trim();
   if (solvtyp == "GK" || solvtyp == "PB")
      eps = 0.1;
   return eps;
}

static FdTestOptions readOptions()
{
   return readFdTestOptions("\n"
                            " Compute the Analytical Gradient Vector [Y] :  ",
      "\n"
      " Compute the Numerical Gradient Vector [Y] :   ",
      defaultFiniteDifferenceStep(), "Ang");
}

static void numericalGradient(std::vector<double>& g, double eps)
{
   g.assign(3 * n, 0);
   for (int i = 0; i < n; ++i) {
      std::array<double*, 3> coord = {{&atoms::x[i], &atoms::y[i], &atoms::z[i]}};
      for (int j = 0; j < 3; ++j) {
         double old = *coord[j];
         *coord[j] = old - 0.5 * eps;
         energy_prec e0 = numericalEnergy();
         *coord[j] = old + 0.5 * eps;
         energy_prec e1 = numericalEnergy();
         *coord[j] = old;
         g[3 * i + j] = (e1 - e0) / eps;
      }
   }

   syncXyzFromHost();
}

void xTestgrad(int, char**)
{
   initial();
   int ixyz;
   tinker_f_getcart(&ixyz);
   tinker_f_mechanic();
   mechanic2();

   auto out = stdout;
   int digits = inform::digits;
   FdTestOptions opts = readOptions();

   int flags = calc::xyz + calc::mass + calc::energy;
   if (opts.analyt)
      flags += calc::grad;

   rc_flag = flags;
   initialize();

   auto fmt = gradientPrintFormat(digits);

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

      energy_prec eval = 0;
      if (opts.analyt) {
         energy(rc_flag);
         copyEnergy(calc::energy, &eval);
      }

      std::vector<double> gdx, gdy, gdz;
      if (opts.analyt) {
         gdx.resize(n);
         gdy.resize(n);
         gdz.resize(n);
         copyGradient(calc::grad, gdx.data(), gdy.data(), gdz.data());
      }
      std::vector<double> ng;
      if (opts.numer)
         numericalGradient(ng, opts.eps);

      if (opts.analyt) {
         const int len_e = 20 + digits;
         const char* fmt_e = "\n Total Potential Energy :%1$*2$.*3$f Kcal/mole\n\n";
         print(out, fmt_e, eval, len_e, digits);
      }

      if (opts.analyt || opts.numer)
         print(out, fmt.header, "");

      for (int i = 0; i < n; ++i) {
         if (opts.analyt)
            printGradientRow(out, fmt.row, "Anlyt", i + 1, gdx[i], gdy[i], gdz[i]);

         if (opts.numer) {
            double nx = getGradientComponent(ng, i, 0);
            double ny = getGradientComponent(ng, i, 1);
            double nz = getGradientComponent(ng, i, 2);
            printGradientRow(out, fmt.row, "Numer", i + 1, nx, ny, nz);
         }
      }

      if (opts.analyt || opts.numer) {
         print(out, "\n\n Total Gradient Norm and RMS Gradient per Atom :\n");
         const char* fmt_summary = "\n %1$s      %2$-30s%3$*4$.*5$f";
         const int len3 = 13 + digits;

         double anlyt_norm = opts.analyt ? totalGradientNorm(gdx, gdy, gdz) : 0;
         double numer_norm = opts.numer ? totalGradientNorm(ng) : 0;
         if (opts.analyt)
            printSummaryRow(out, fmt_summary, "Anlyt", "Total Gradient Norm Value", anlyt_norm, len3, digits);
         if (opts.numer)
            printSummaryRow(out, fmt_summary, "Numer", "Total Gradient Norm Value", numer_norm, len3, digits);
         print(out, "\n");

         if (opts.analyt)
            printSummaryRow(out, fmt_summary, "Anlyt", "RMS Gradient over All Atoms", anlyt_norm / std::sqrt(n), len3,
               digits);
         if (opts.numer)
            printSummaryRow(out, fmt_summary, "Numer", "RMS Gradient over All Atoms", numer_norm / std::sqrt(n), len3,
               digits);
         print(out, "\n");
      }
   } while (not done);

   finish();
   tinker_f_final();
}
}
