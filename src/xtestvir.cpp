#include "ff/atom.h"
#include "ff/box.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "tool/ioprint.h"
#include <array>
#include <cmath>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/boxes.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>
#include <vector>

#include "tinker9.h"

namespace tinker {
static constexpr double finite_difference_eps = 0.02;

static double& lvec(int row, int col)
{
   return boxes::lvec[col][row];
}

static double& recip(int row, int col)
{
   return boxes::recip[col][row];
}

static void printMatrix(FILE* out, const char* title, int nspace, const double (&m)[3][3])
{
   print(out, "\n %s :%*s%13.3f%13.3f%13.3f", title, nspace, "", m[0][0], m[0][1], m[0][2]);
   print(out, "\n%36s%13.3f%13.3f%13.3f", "", m[1][0], m[1][1], m[1][2]);
   print(out, "\n%36s%13.3f%13.3f%13.3f\n", "", m[2][0], m[2][1], m[2][2]);
}

static void syncBoxAndXyz()
{
   boxData(RcOp::INIT);
   xyzData(RcOp::INIT);
   nblistRefresh();
}

static energy_prec numericalEnergy()
{
   syncBoxAndXyz();
   energy(calc::energy);
   energy_prec eout;
   copyEnergy(calc::energy, &eout);
   return eout;
}

static void cellang(const std::vector<double>& xf, const std::vector<double>& yf, const std::vector<double>& zf)
{
   for (int i = 0; i < n; ++i) {
      atoms::x[i] = xf[i] * lvec(0, 0) + yf[i] * lvec(1, 0) + zf[i] * lvec(2, 0);
      atoms::y[i] = xf[i] * lvec(0, 1) + yf[i] * lvec(1, 1) + zf[i] * lvec(2, 1);
      atoms::z[i] = xf[i] * lvec(0, 2) + yf[i] * lvec(1, 2) + zf[i] * lvec(2, 2);
   }

   double amag = std::sqrt(lvec(0, 0) * lvec(0, 0) + lvec(0, 1) * lvec(0, 1) + lvec(0, 2) * lvec(0, 2));
   double bmag = std::sqrt(lvec(1, 0) * lvec(1, 0) + lvec(1, 1) * lvec(1, 1) + lvec(1, 2) * lvec(1, 2));
   double cmag = std::sqrt(lvec(2, 0) * lvec(2, 0) + lvec(2, 1) * lvec(2, 1) + lvec(2, 2) * lvec(2, 2));
   double abdot = lvec(0, 0) * lvec(1, 0) + lvec(0, 1) * lvec(1, 1) + lvec(0, 2) * lvec(1, 2);
   double acdot = lvec(0, 0) * lvec(2, 0) + lvec(0, 1) * lvec(2, 1) + lvec(0, 2) * lvec(2, 2);
   double bcdot = lvec(1, 0) * lvec(2, 0) + lvec(1, 1) * lvec(2, 1) + lvec(1, 2) * lvec(2, 2);

   boxes::xbox = amag;
   boxes::ybox = bmag;
   boxes::zbox = cmag;
   boxes::alpha = (180 / M_PI) * std::acos(bcdot / (bmag * cmag));
   boxes::beta = (180 / M_PI) * std::acos(acdot / (amag * cmag));
   boxes::gamma = (180 / M_PI) * std::acos(abdot / (amag * bmag));
   tinker_f_lattice();
}

static std::array<std::vector<double>, 3> fractionalCoordinates()
{
   std::array<std::vector<double>, 3> frac = {{{}, {}, {}}};
   for (auto& f : frac)
      f.resize(n);

   for (int i = 0; i < n; ++i) {
      frac[0][i] = atoms::x[i] * recip(0, 0) + atoms::y[i] * recip(1, 0) + atoms::z[i] * recip(2, 0);
      frac[1][i] = atoms::x[i] * recip(0, 1) + atoms::y[i] * recip(1, 1) + atoms::z[i] * recip(2, 1);
      frac[2][i] = atoms::x[i] * recip(0, 2) + atoms::y[i] * recip(1, 2) + atoms::z[i] * recip(2, 2);
   }

   return frac;
}

static void finiteDifferenceLvec(double (&dedl)[3][3], const std::array<std::vector<double>, 3>& frac)
{
   for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
         dedl[j][i] = 0;

   for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
         double old = lvec(j, i);
         lvec(j, i) = old - finite_difference_eps;
         cellang(frac[0], frac[1], frac[2]);
         energy_prec eneg = numericalEnergy();

         lvec(j, i) = old + finite_difference_eps;
         cellang(frac[0], frac[1], frac[2]);
         energy_prec epos = numericalEnergy();

         lvec(j, i) = old;
         cellang(frac[0], frac[1], frac[2]);
         dedl[j][i] = 0.5 * (epos - eneg) / finite_difference_eps;
      }
   }
}

static void numericalVirial(double (&virn)[3][3], const double (&dedl)[3][3])
{
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j <= i; ++j) {
         virn[j][i] = 0;
         for (int k = 0; k < 3; ++k)
            virn[j][i] += dedl[k][j] * lvec(k, i);
         virn[i][j] = virn[j][i];
      }
   }
}

static void ptest(FILE* out)
{
   if (!bound::use_bounds)
      return;

   if (!boxes::nonprism) {
      boxes::orthogonal = 0;
      boxes::monoclinic = 0;
      boxes::triclinic = 1;
   }

   double lmat[3][3] = {
      {lvec(0, 0), lvec(1, 0), lvec(2, 0)},
      {lvec(0, 1), lvec(1, 1), lvec(2, 1)},
      {lvec(0, 2), lvec(1, 2), lvec(2, 2)},
   };
   printMatrix(out, "Lattice Vectors (Lvec)", 11, lmat);

   auto frac = fractionalCoordinates();

   double dedl[3][3] = {};
   finiteDifferenceLvec(dedl, frac);
   double dedl_print[3][3] = {
      {dedl[0][0], dedl[1][0], dedl[2][0]},
      {dedl[0][1], dedl[1][1], dedl[2][1]},
      {dedl[0][2], dedl[1][2], dedl[2][2]},
   };
   printMatrix(out, "dE/dLvec Derivatives", 13, dedl_print);

   double virn[3][3] = {};
   numericalVirial(virn, dedl);
   if (boxes::dodecadron) {
      print(out, "\n Numerical Mean Diagonal :%10s%13.3f\n", "", (virn[0][0] + virn[1][1] + virn[2][2]) / 3.0);
   } else if (boxes::octahedron) {
      print(out, "\n Numerical Virial Diagonal :%8s%13.3f%13.3f%13.3f\n", "", virn[0][0], virn[1][1], virn[2][2]);
   } else {
      double virn_print[3][3] = {
         {virn[0][0], virn[1][0], virn[2][0]},
         {virn[0][1], virn[1][1], virn[2][1]},
         {virn[0][2], virn[1][2], virn[2][2]},
      };
      printMatrix(out, "Numerical Virial Tensor", 10, virn_print);
   }

   double dedv_vir = (vir[0] + vir[4] + vir[8]) / (3.0 * boxes::volbox);
   double dedv_num = (virn[0][0] + virn[1][1] + virn[2][2]) / (3.0 * boxes::volbox);
   double temp = bath::kelvin;
   if (temp == 0)
      temp = 298;
   double pres_vir = units::prescon * (static_cast<double>(n) * units::gasconst * temp / boxes::volbox - dedv_vir);
   double pres_num = units::prescon * (static_cast<double>(n) * units::gasconst * temp / boxes::volbox - dedv_num);
   print(out, "\n Pressure (Analytical,%4d K) :%5s%13.3f Atmospheres\n", static_cast<int>(std::round(temp)), "",
      pres_vir);
   print(out, " Pressure (Numerical,%4d K) :%6s%13.3f Atmospheres\n", static_cast<int>(std::round(temp)), "", pres_num);
}

void xTestvir(int, char**)
{
   initial();
   tinker_f_getxyz();
   tinker_f_mechanic();
   mechanic2();

   inform::debug = 0;
   rc_flag = calc::xyz + calc::energy + calc::grad + calc::virial;
   initialize();

   energy(rc_flag);
   double analytic[3][3] = {
      {vir[0], vir[1], vir[2]},
      {vir[3], vir[4], vir[5]},
      {vir[6], vir[7], vir[8]},
   };
   printMatrix(stdout, "Analytical Virial Tensor", 9, analytic);

   ptest(stdout);

   finish();
   tinker_f_final();
}
}
