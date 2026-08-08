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

int testgradFlags(const FdTestOptions& opts)
{
   int flags = calc::xyz + calc::mass + calc::energy;
   if (opts.analyt)
      flags += calc::grad;
   return flags;
}

TestgradResult testgradEvaluate(const FdTestOptions& opts)
{
   TestgradResult r;

   if (opts.analyt) {
      energy(rc_flag);
      copyEnergy(calc::energy, &r.energy);

      copyGradientFlat(calc::grad, r.ganlyt);
   }

   if (opts.numer)
      numericalGradient(r.gnumer, opts.eps);

   return r;
}

void testgradPrint(FILE* out, const FdTestOptions& opts, const TestgradResult& r, int digits)
{
   auto fmt = gradientPrintFormat(digits);

   if (opts.analyt) {
      const int len_e = 20 + digits;
      const char* fmt_e = "\n Total Potential Energy :%1$*2$.*3$f Kcal/mole\n\n";
      print(out, fmt_e, r.energy, len_e, digits);
   }

   printGradientTable(out, "Cartesian Gradient Breakdown over Individual Atoms", fmt, opts, r.ganlyt, r.gnumer, digits,
      std::sqrt(n));
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

   rc_flag = testgradFlags(opts);
   initialize();

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

      testgradPrint(out, opts, testgradEvaluate(opts), digits);
   } while (not done);

   finish();
   tinker_f_final();
}
}
