#include "ff/energy.h"
#include "ff/nblist.h"
#include "md/rattle.h"
#include "tool/argkey.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/ioread.h"
#include <tinker/detail/atoms.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/scales.hh>
#include <tinker/routines.h>

#include "tinker9.h"

namespace tinker {
static std::vector<double> grx, gry, grz;

TINKER_FVOID2(acc1, cu1, xMinimizeSetPos, int, const double*, const double*);
static void xMinimizeSetXyz(int n, const double* xx, const double* scale)
{
   TINKER_FCALL2(acc1, cu1, xMinimizeSetPos, n, xx, scale);
   copyPosToXyz();
   nblistRefresh();
}

TINKER_FVOID2(acc1, cu1, xMinimizeSetXxByPos, int, double*, const double*);
static void xMinimizeSetXx(int n, double* xx, const double* scale)
{
   TINKER_FCALL2(acc1, cu1, xMinimizeSetXxByPos, n, xx, scale);
}

TINKER_FVOID2(acc1, cu1, xMinimizeEnd);

static void xMinimizeSetHostXyz(int n, const double* xx, const double* scale)
{
   for (int i = 0; i < n; ++i) {
      int ii = 3 * i;
      atoms::x[i] = xx[ii + 0] / scale[ii + 0];
      atoms::y[i] = xx[ii + 1] / scale[ii + 1];
      atoms::z[i] = xx[ii + 2] / scale[ii + 2];
   }
}

// As in minimize.f, shake moves the atoms back onto the distance constraints,
// rigid waters included, and places the extra sites of four-site waters.
static void xMinimizeShake(int n, double* xx, const double* scale)
{
   xMinimizeSetHostXyz(n, xx, scale);
   tinker_f_shake(atoms::x, atoms::y, atoms::z);
   for (int i = 0; i < n; ++i) {
      int ii = 3 * i;
      xx[ii + 0] = atoms::x[i] * scale[ii + 0];
      xx[ii + 1] = atoms::y[i] * scale[ii + 1];
      xx[ii + 2] = atoms::z[i] * scale[ii + 2];
   }
}

// Interleaves the gradient into g; as in minimize.f, shakeg then removes the
// components along the distance constraints, at the coordinates in atoms::x.
static void xMinimizeGradient(int n, double* g, bool userat)
{
   for (int i = 0; i < n; ++i) {
      g[3 * i + 0] = grx[i];
      g[3 * i + 1] = gry[i];
      g[3 * i + 2] = grz[i];
   }
   if (userat)
      tinker_f_shakeg(g);
}

double minimizeFunction(double* xx, double* g)
{
   // convert optimization parameters to atomic coordinates; xx keeps the
   // coordinates as shake left them
   const bool userat = useRattle();
   if (userat)
      xMinimizeShake(n, xx, scales::scale);
   xMinimizeSetXyz(n, xx, scales::scale);

   // compute and store the energy and gradient
   energy(calc::energy + calc::grad);
   energy_prec eout;
   copyEnergy(calc::energy, &eout);
   copyGradient(calc::grad, grx.data(), gry.data(), grz.data());

   // convert gradient to optimization parameters
   xMinimizeGradient(n, g, userat);
   for (int i = 0; i < 3 * n; ++i)
      g[i] /= scales::scale[i];

   return eout;
}

std::vector<double> minimizeParams()
{
   // perform dynamic allocation of some global arrays
   if (!tinker_f_allocated(scales::scale))
      tinker_f_allocate_element(&scales::scale, 3 * n);

   // set scaling parameter for function and derivative values;
   // use square root of median eigenvalue of typical Hessian
   scales::set_scale = 1;
   int nvar = 0;
   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j) {
         scales::scale[nvar] = 12;
         ++nvar;
      }
   }

   // perform dynamic allocation of some local arrays
   std::vector<double> xxvec(3 * n);
   grx.resize(n);
   gry.resize(n);
   grz.resize(n);

   // convert atomic coordinates to optimization parameters
   xMinimizeSetXx(n, xxvec.data(), scales::scale);
   return xxvec;
}

void minimizeEnd()
{
   grx.clear();
   gry.clear();
   grz.clear();
   TINKER_FCALL2(acc1, cu1, xMinimizeEnd);
}
}

namespace tinker {
void xMinimize(int, char**)
{
   initial();
   tinker_f_getxyz();
   tinker_f_mechanic();
   mechanic2();

   // perform the setup functions needed for optimization
   tinker_f_optinit();

   // get termination criterion as RMS gradient per atom
   bool exist = false;
   char string[240];
   double grdmin = -1.0;
   nextarg(string, exist);
   if (exist) {
      ioReadString(grdmin, string);
   }
   std::string prompt = "\n"
                        " Enter RMS Gradient per Atom Criterion [0.01] :  ";
   ioReadStream(grdmin, prompt, 0.01, [](double val) { return val < 0; });

   // write out a copy of coordinates for later update
   int imin = tinker_f_freeunit();
   const int leng = files::leng;
   std::string minfile = FstrView(files::filename)(1, leng).trim() + ".xyz";
   minfile = tinker_f_version(minfile, "new");
   tinker_f_open(&imin, minfile, "new");
   tinker_f_prtxyz(&imin);
   tinker_f_close(&imin);
   FstrView outview = files::outfile;
   outview = minfile;

   int flags = calc::xyz + calc::mass;
   flags += (calc::energy + calc::grad);
   rc_flag = flags;
   initialize();

   std::vector<double> xxvec = minimizeParams();
   double* xx = xxvec.data();

   // make the call to the optimization routine
   int n3 = 3 * n;
   double mini;
   tinker_f_lbfgs(&n3, xx, &mini, &grdmin, minimizeFunction, tinker_f_optsave);

   // convert optimization parameters to atomic coordinates
   xMinimizeSetXyz(n, xx, scales::scale);

   // compute the final function and RMS gradient values
   double minimum;
   energy(calc::energy + calc::grad);
   copyEnergy(calc::energy, &minimum);
   copyGradient(calc::grad, grx.data(), gry.data(), grz.data());
   const bool userat = useRattle();
   if (userat)
      xMinimizeSetHostXyz(n, xx, scales::scale);
   std::vector<double> gvec(n3);
   xMinimizeGradient(n, gvec.data(), userat);
   double gnorm = 0;
   for (int i = 0; i < n3; ++i) {
      gnorm += gvec[i] * gvec[i];
   }
   gnorm = std::sqrt(gnorm);
   double grms = gnorm / std::sqrt(n);

   // perform deallocation of some local arrays
   xxvec.clear();
   minimizeEnd();

   // write out the final function and gradient values
   auto o = stdout;
   const int d1 = inform::digits;
   int l1 = 12 + d1;
   auto fstr = "%1$*2$.*3$f"_s;
   auto estr = fstr;
   if (grms <= std::pow(10, -d1))
      estr = "%1$*2$.*3$E"_s;
   print(o, "\n Final Function Value :  "_s + fstr, minimum, l1, d1);
   print(o, "\n Final RMS Gradient :    "_s + estr, grms, l1, d1);
   print(o, "\n Final Gradient Norm :   "_s + estr, gnorm, l1, d1);
   print(o, "\n");

   // write the final coordinates into a file
   bounds();
   imin = tinker_f_freeunit();
   tinker_f_open(&imin, minfile, "old");
   tinker_f_rewind(&imin);
   tinker_f_prtxyz(&imin);
   tinker_f_close(&imin);

   finish();
   tinker_f_final();
}
}
