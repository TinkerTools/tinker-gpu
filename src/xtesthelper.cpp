#include "tool/xtesthelper.h"

#include "ff/atom.h"
#include "ff/egvop.h"
#include "ff/nblist.h"
#include "tool/darray.h"
#include "tool/ioprint.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <tinker/detail/atoms.hh>

namespace tinker {
static bool invalidYesNo(char c)
{
   return c != 'Y' && c != 'y' && c != 'N' && c != 'n';
}

static bool answerIsNo(char c)
{
   return c == 'N' || c == 'n';
}

bool askYesNo(const std::string& prompt, char dflt)
{
   char answer = askValue<char>(prompt, ' ', dflt, invalidYesNo);
   return !answerIsNo(answer);
}

std::string compactReal(double value)
{
   std::ostringstream os;
   os << value;
   return os.str();
}

double askFiniteDifferenceStep(double dfltEps, const char* unit)
{
   return askValue<double>("\n"
                           " Enter Finite Difference Stepsize ["
         + compactReal(dfltEps) + " " + unit + "] :  ",
      -1.0, dfltEps, [](double val) { return val <= 0; });
}

FdTestOptions readFdTestOptions(const std::string& analytPrompt, const std::string& numerPrompt, double dfltEps,
   const char* unit)
{
   FdTestOptions opts;
   opts.analyt = askYesNo(analytPrompt);
   opts.numer = askYesNo(numerPrompt);
   if (opts.numer)
      opts.eps = askFiniteDifferenceStep(dfltEps, unit);
   return opts;
}

void syncXyzFromHost()
{
   darray::copyin(g::q0, n, xpos, atoms::x);
   darray::copyin(g::q0, n, ypos, atoms::y);
   darray::copyin(g::q0, n, zpos, atoms::z);
   copyPosToXyz();
   nblistRefresh();
}

energy_prec evaluateEnergy()
{
   energy(calc::energy);
   energy_prec eout;
   copyEnergy(calc::energy, &eout);
   return eout;
}

static void interleave(std::vector<double>& g, const std::vector<double>& gx, const std::vector<double>& gy,
   const std::vector<double>& gz)
{
   g.resize(3 * n);
   for (int i = 0; i < n; ++i) {
      g[3 * i + 0] = gx[i];
      g[3 * i + 1] = gy[i];
      g[3 * i + 2] = gz[i];
   }
}

void copyGradientFlat(int vers, std::vector<double>& g)
{
   std::vector<double> gx(n), gy(n), gz(n);
   copyGradient(vers, gx.data(), gy.data(), gz.data());
   interleave(g, gx, gy, gz);
}

void copyGradientFlat(int vers, std::vector<double>& g, const grad_prec* gxSrc, const grad_prec* gySrc,
   const grad_prec* gzSrc)
{
   std::vector<double> gx(n), gy(n), gz(n);
   copyGradient(vers, gx.data(), gy.data(), gz.data(), gxSrc, gySrc, gzSrc);
   interleave(g, gx, gy, gz);
}

double vectorNorm(double x, double y, double z)
{
   return std::sqrt(x * x + y * y + z * z);
}

double getGradientComponent(const std::vector<double>& g, int i, int j)
{
   return g[3 * i + j];
}

double totalGradientNorm(const std::vector<double>& g)
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

// Emits "%1$<pad>s", the positional-argument padding used by the table headers. The pad shrinks
// by however much the label that follows it exceeds the historical 5-character width, so every
// column stays at a fixed position regardless of the label text.
static std::string headerPad(int base, const char* label)
{
   int pad = base - (static_cast<int>(std::strlen(label)) - 5);
   if (pad < 1)
      pad = 1;
   return "%1$" + std::to_string(pad) + "s";
}

GradientPrintFormat gradientPrintFormat(int digits, const char* cx, const char* cy, const char* cz)
{
   const char* prefix;
   int b1, b2, b3, b4;
   std::string row;
   if (digits == 8) {
      prefix = "\n  Type    Atom ";
      b1 = 8, b2 = 9, b3 = 9, b4 = 9;
      row = "\n %s%8d %16.8f%16.8f%16.8f%16.8f";
   } else if (digits == 6) {
      prefix = "\n  Type      Atom ";
      b1 = 9, b2 = 7, b3 = 7, b4 = 9;
      row = "\n %s%10d   %14.6f%14.6f%14.6f  %14.6f";
   } else {
      prefix = "\n  Type      Atom ";
      b1 = 12, b2 = 5, b3 = 5, b4 = 8;
      row = "\n %s%10d       %12.4f%12.4f%12.4f  %12.4f";
   }

   std::string header = prefix;
   header += headerPad(b1, cx) + " " + cx + " ";
   header += headerPad(b2, cy) + " " + cy + " ";
   header += headerPad(b3, cz) + " " + cz + " ";
   // The trailing "Norm" label is fixed, so its pad keeps its base width; the
   // preceding pads have already absorbed any change in the column labels.
   header += "%1$" + std::to_string(b4) + "s Norm\n";
   return {header, row};
}

void printGradientRow(FILE* out, const std::string& fmt, const char* label, int atom, double gx, double gy,
   double gz)
{
   print(out, fmt, label, atom, gx, gy, gz, vectorNorm(gx, gy, gz));
}

void printSummaryRow(FILE* out, const char* fmt, const char* label, const char* title, double value, int width,
   int digits)
{
   print(out, fmt, label, title, value, width, digits);
}

void printGradientTable(FILE* out, const char* title, const GradientPrintFormat& fmt, const FdTestOptions& opts,
   const std::vector<double>& anlyt, const std::vector<double>& numer, int digits, double rmsDenom)
{
   if (not(opts.analyt or opts.numer))
      return;

   print(out, "\n %s :\n", title);
   print(out, fmt.header, "");

   for (int i = 0; i < n; ++i) {
      if (opts.analyt)
         printGradientRow(out, fmt.row, "Anlyt", i + 1, getGradientComponent(anlyt, i, 0),
            getGradientComponent(anlyt, i, 1), getGradientComponent(anlyt, i, 2));
      if (opts.numer)
         printGradientRow(out, fmt.row, "Numer", i + 1, getGradientComponent(numer, i, 0),
            getGradientComponent(numer, i, 1), getGradientComponent(numer, i, 2));
   }

   print(out, "\n\n Total Gradient Norm and RMS Gradient per Atom :\n");
   const char* fmtSummary = "\n %1$s      %2$-30s%3$*4$.*5$f";
   const int width = 13 + digits;

   double anlytNorm = opts.analyt ? totalGradientNorm(anlyt) : 0;
   double numerNorm = opts.numer ? totalGradientNorm(numer) : 0;
   if (opts.analyt)
      printSummaryRow(out, fmtSummary, "Anlyt", "Total Gradient Norm Value", anlytNorm, width, digits);
   if (opts.numer)
      printSummaryRow(out, fmtSummary, "Numer", "Total Gradient Norm Value", numerNorm, width, digits);
   print(out, "\n");

   if (opts.analyt)
      printSummaryRow(out, fmtSummary, "Anlyt", "RMS Gradient over All Atoms", anlytNorm / rmsDenom, width, digits);
   if (opts.numer)
      printSummaryRow(out, fmtSummary, "Numer", "RMS Gradient over All Atoms", numerNorm / rmsDenom, width, digits);
   print(out, "\n");
}

void printMatrix(FILE* out, const char* title, int nspace, const double (&m)[3][3], int indent)
{
   print(out, "\n %s :%*s%13.3f%13.3f%13.3f", title, nspace, "", m[0][0], m[0][1], m[0][2]);
   print(out, "\n%*s%13.3f%13.3f%13.3f", indent, "", m[1][0], m[1][1], m[1][2]);
   print(out, "\n%*s%13.3f%13.3f%13.3f\n", indent, "", m[2][0], m[2][1], m[2][2]);
}

void printMatrix(FILE* out, const char* title, int nspace, const double* m9, int indent)
{
   const double m[3][3] = {
      {m9[0], m9[1], m9[2]},
      {m9[3], m9[4], m9[5]},
      {m9[6], m9[7], m9[8]},
   };
   printMatrix(out, title, nspace, m, indent);
}
}
