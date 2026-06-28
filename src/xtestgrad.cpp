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
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/solpot.hh>
#include <tinker/routines.h>
#include <vector>

#include "tinker9.h"

namespace tinker {
static void syncXyzFromHost()
{
   darray::copyin(g::q0, n, xpos, atoms::x);
   darray::copyin(g::q0, n, ypos, atoms::y);
   darray::copyin(g::q0, n, zpos, atoms::z);
   copyPosToXyz();
   nblistRefresh();
}

static energy_prec numericalEnergy()
{
   syncXyzFromHost();
   energy(calc::energy);
   energy_prec eout;
   copyEnergy(calc::energy, &eout);
   return eout;
}

static double getGradientComponent(const std::vector<double>& g, int i, int j)
{
   return g[3 * i + j];
}

struct TestgradOptions
{
   bool analyt = true;
   bool numer = true;
   double eps = 0;
};

struct GradientPrintFormat
{
   std::string header;
   std::string row;
};

static bool invalidYesNo(char c)
{
   return c != 'Y' && c != 'y' && c != 'N' && c != 'n';
}

static bool answerIsNo(char c)
{
   return c == 'N' || c == 'n';
}

template <size_t Len>
static bool readYesNo(const std::string& prompt, char (&buffer)[Len], bool& exist)
{
   char answer = ' ';
   nextarg(buffer, exist);
   if (exist)
      ioReadString(answer, buffer);
   ioReadStream(answer, prompt, 'Y', invalidYesNo);
   return !answerIsNo(answer);
}

static double defaultFiniteDifferenceStep()
{
   double eps = 0.01;
   std::string solvtyp = FstrView(solpot::solvtyp).trim();
   if (solvtyp == "GK" || solvtyp == "PB")
      eps = 0.1;
   return eps;
}

static std::string compactReal(double value)
{
   std::ostringstream os;
   os << value;
   return os.str();
}

template <size_t Len>
static double readFiniteDifferenceStep(double default_eps, char (&buffer)[Len], bool& exist)
{
   double eps = -1;
   nextarg(buffer, exist);
   if (exist)
      ioReadString(eps, buffer);

   ioReadStream(eps,
      "\n"
      " Enter Finite Difference Stepsize ["
         + compactReal(default_eps) + " Ang] :  ",
      default_eps, [](double val) { return val <= 0; });
   return eps;
}

static TestgradOptions readOptions()
{
   bool exist = false;
   char buffer[240];
   TestgradOptions opts;

   opts.analyt = readYesNo("\n"
                           " Compute the Analytical Gradient Vector [Y] :  ",
      buffer, exist);
   opts.numer = readYesNo("\n"
                          " Compute the Numerical Gradient Vector [Y] :   ",
      buffer, exist);
   if (opts.numer)
      opts.eps = readFiniteDifferenceStep(defaultFiniteDifferenceStep(), buffer, exist);

   return opts;
}

static GradientPrintFormat gradientPrintFormat(int digits)
{
   if (digits == 8) {
      return {"\n  Type    Atom %1$8s "
              "dE/dX %1$9s dE/dY %1$9s dE/dZ %1$9s Norm\n",
         "\n %s%8d %16.8f%16.8f%16.8f%16.8f"};
   } else if (digits == 6) {
      return {"\n  Type      Atom %1$9s "
              "dE/dX %1$7s dE/dY %1$7s dE/dZ %1$9s Norm\n",
         "\n %s%10d   %14.6f%14.6f%14.6f  %14.6f"};
   } else {
      return {"\n  Type      Atom %1$12s "
              "dE/dX %1$5s dE/dY %1$5s dE/dZ %1$8s Norm\n",
         "\n %s%10d       %12.4f%12.4f%12.4f  %12.4f"};
   }
}

static double vectorNorm(double x, double y, double z)
{
   return std::sqrt(x * x + y * y + z * z);
}

static double totalGradientNorm(const std::vector<double>& gx, const std::vector<double>& gy,
   const std::vector<double>& gz)
{
   double norm2 = 0;
   for (int i = 0; i < n; ++i)
      norm2 += gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
   return std::sqrt(norm2);
}

static double totalGradientNorm(const std::vector<double>& g)
{
   double norm2 = 0;
   for (int i = 0; i < n; ++i) {
      double gx = getGradientComponent(g, i, 0);
      double gy = getGradientComponent(g, i, 1);
      double gz = getGradientComponent(g, i, 2);
      norm2 += gx * gx + gy * gy + gz * gz;
   }
   return std::sqrt(norm2);
}

static void printGradientRow(FILE* out, const std::string& fmt, const char* label, int atom, double gx, double gy,
   double gz)
{
   print(out, fmt, label, atom, gx, gy, gz, vectorNorm(gx, gy, gz));
}

static void printSummaryRow(FILE* out, const char* fmt, const char* label, const char* title, double value, int width,
   int digits)
{
   print(out, fmt, label, title, value, width, digits);
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
   TestgradOptions opts = readOptions();

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
      // auto do_print = [](int i, int n, int top_m) {
      //    if (n <= 2 * top_m)
      //       return true;
      //    else if (i < top_m)
      //       return true;
      //    else if (i >= n - top_m)
      //       return true;
      //    else
      //       return false;
      // };
      // int print_top_n = 15;
      for (int i = 0; i < n; ++i) {
         // if (not do_print(i, n, print_top_n))
         //    continue;

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
