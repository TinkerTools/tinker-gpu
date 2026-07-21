#pragma once
#include "ff/energy.h"
#include <cstdio>
#include <sstream> // tool/ioread.h uses std::istringstream without including <sstream>
#include <string>
#include <vector>

#include "tool/argkey.h"
#include "tool/ioread.h"

namespace tinker {
/// \addtogroup general
/// \{
///
/// \file
/// Helpers shared by the \c x*test* driver programs (\c xtestgrad, \c xtestlmda, \c xtestvir).
///
/// \note Unrelated to \c include/test.h and \c include/testrt.h, which belong to the Catch2
/// unit-test suite under \c test/.

//====================================================================//
//                        Interactive Prompting                       //
//====================================================================//

/// Reads one value: consumes the next command-line argument if one is present,
/// otherwise prompts on stdin.
///
/// \param prompt    Help message sent to stdout when a prompt is needed.
/// \param initial   Seed value. It \b must satisfy <tt>invalid(initial) == true</tt>, otherwise
///                  no prompt is ever issued (#ioReadStream only prompts for invalid values).
/// \param autoFill  Value assigned when the user submits an empty line.
/// \param invalid   Predicate returning true when a value is out of range.
template <class T, class Invalid>
T askValue(const std::string& prompt, T initial, T autoFill, Invalid&& invalid)
{
   bool exist = false;
   char buffer[240];
   T value = initial;
   nextarg(buffer, exist);
   if (exist)
      ioReadString(value, buffer);
   ioReadStream(value, prompt, autoFill, invalid);
   return value;
}

/// Asks a yes/no question. Returns true unless the answer is N or n.
bool askYesNo(const std::string& prompt, char dflt = 'Y');

/// Formats a real number using the shortest default representation.
std::string compactReal(double value);

/// Asks for a finite difference stepsize, e.g. \c unit is "Ang" or "Lambda".
double askFiniteDifferenceStep(double dfltEps, const char* unit);

/// Options shared by the analytical-vs-numerical test programs.
struct FdTestOptions
{
   bool analyt = true;
   bool numer = true;
   double eps = 0;
};

/// Asks whether to compute the analytical and/or numerical values, then the stepsize.
FdTestOptions readFdTestOptions(const std::string& analytPrompt, const std::string& numerPrompt, double dfltEps,
   const char* unit);

//====================================================================//
//                         Energy Evaluation                          //
//====================================================================//

/// Copies the host coordinates (\c atoms::x/y/z) to the device and refreshes the neighbor lists.
void syncXyzFromHost();

/// Evaluates the potential energy only and returns it on the host.
energy_prec evaluateEnergy();

//====================================================================//
//                               Norms                                //
//====================================================================//

/// Returns the length of a 3-vector.
double vectorNorm(double x, double y, double z);

/// Returns component \c j of atom \c i from an interleaved 3n gradient array.
double getGradientComponent(const std::vector<double>& g, int i, int j);

/// Returns the norm of a gradient stored as three n-length arrays.
double totalGradientNorm(const std::vector<double>& gx, const std::vector<double>& gy,
   const std::vector<double>& gz);

/// Returns the norm of a gradient stored as one interleaved 3n array.
double totalGradientNorm(const std::vector<double>& g);

//====================================================================//
//                              Printing                              //
//====================================================================//

/// Header and per-row format strings for a gradient breakdown table.
struct GradientPrintFormat
{
   std::string header;
   std::string row;
};

/// Builds the format strings for a per-atom gradient table at the given precision.
///
/// The column labels are configurable; the padding preceding each label absorbs that label's
/// length so the columns stay at fixed positions. With the default labels the result is
/// byte-for-byte identical to the historical \c xtestgrad format strings.
GradientPrintFormat gradientPrintFormat(int digits, const char* cx = "dE/dX", const char* cy = "dE/dY",
   const char* cz = "dE/dZ");

/// Prints one row of a gradient table, appending the vector norm.
void printGradientRow(FILE* out, const std::string& fmt, const char* label, int atom, double gx, double gy,
   double gz);

/// Prints one labeled summary line, e.g. a total norm or an RMS value.
void printSummaryRow(FILE* out, const char* fmt, const char* label, const char* title, double value, int width,
   int digits);

/// Prints a titled 3x3 matrix as <tt>' <title> :'</tt>, \c nspace blanks, then the first row;
/// the remaining two rows are indented by \c indent columns.
void printMatrix(FILE* out, const char* title, int nspace, const double (&m)[3][3], int indent = 36);

/// Prints a titled 3x3 matrix stored as a row-major flat array of 9 elements.
void printMatrix(FILE* out, const char* title, int nspace, const double* m9, int indent = 36);
/// \}
}
