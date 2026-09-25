#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "math/random.h"
#include "md/rattle.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace tinker {
static bool s_on = false;       // keyword INDUCE-WIGGLE in MD
static bool s_failed = false;   // a solver failed in this energy evaluation
static bool s_retrying = false; // atoms were wiggled; disable the predictor
static int s_ntry = 0;          // wiggles for the current energy evaluation

void induceWiggleData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      s_on = false;
      s_failed = false;
      s_retrying = false;
      s_ntry = 0;
   }

   if (op & RcOp::INIT) {
      bool kw;
      getKV("INDUCE-WIGGLE", kw, false);
      s_on = kw and (rc_flag & calc::md);
   }
}

bool induceWiggleOn()
{
   return s_on;
}

bool induceNoPredict()
{
   return s_retrying;
}

bool induceFailed()
{
   return s_failed;
}

void induceReportFailure(const real (*rsd)[3], const real (*rsdp)[3])
{
   std::vector<real> r(3 * n), rp;
   darray::copyout(g::q0, n, r.data(), rsd);
   if (rsdp) {
      rp.resize(3 * n);
      darray::copyout(g::q0, n, rp.data(), rsdp);
   }
   waitFor(g::q0);

   std::vector<double> rsq(n);
   std::vector<int> idx(n);
   for (int i = 0; i < n; ++i) {
      double s = 0, sp = 0;
      for (int j = 0; j < 3; ++j) {
         s += r[3 * i + j] * r[3 * i + j];
         if (rsdp)
            sp += rp[3 * i + j] * rp[3 * i + j];
      }
      rsq[i] = std::max(s, sp);
      idx[i] = i;
   }
   int ntop = std::min(n, 5);
   std::partial_sort(idx.begin(), idx.begin() + ntop, idx.end(),
      [&](int a, int b) { return rsq[a] > rsq[b]; });

   print(stdout, " INDUCE  --  Warning, Induced Dipoles are not Converged\n");
   print(stdout, " Largest Squared Residuals :\n");
   for (int k = 0; k < ntop; ++k)
      print(stdout, "    Atom %8d     Residual %12.4e\n", idx[k] + 1, rsq[idx[k]]);

   s_failed = true;
}

bool induceWiggleRetry()
{
   if (not s_failed) {
      s_ntry = 0;
      s_retrying = false;
      return false;
   }

   constexpr int maxtry = 5;
   constexpr double deltas[maxtry] = {1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3};
   if (s_ntry >= maxtry) {
      s_failed = false;
      s_retrying = false;
      s_ntry = 0;
      printError();
      TINKER_THROW(format("INDUCE  --  Warning, Induced Dipoles are not Converged after %d Wiggle Attempts", maxtry));
   }

   double delta = deltas[s_ntry];
   print(stdout, " Trying Wiggle by %.2e Ang (Attempt %d of %d)\n", delta, s_ntry + 1, maxtry);

   std::vector<pos_prec> x0(n), y0(n), z0(n), x1(n), y1(n), z1(n);
   darray::copyout(g::q0, n, x0.data(), xpos);
   darray::copyout(g::q0, n, y0.data(), ypos);
   darray::copyout(g::q0, n, z0.data(), zpos);
   waitFor(g::q0);
   for (int i = 0; i < n; ++i) {
      double u[3], unorm;
      do {
         u[0] = normal<double>();
         u[1] = normal<double>();
         u[2] = normal<double>();
         unorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      } while (unorm == 0);
      x1[i] = x0[i] + delta * u[0] / unorm;
      y1[i] = y0[i] + delta * u[1] / unorm;
      z1[i] = z0[i] + delta * u[2] / unorm;
   }
   darray::copyin(g::q0, n, xpos, x1.data());
   darray::copyin(g::q0, n, ypos, y1.data());
   darray::copyin(g::q0, n, zpos, z1.data());

   // restore holonomic constraints with respect to the unwiggled positions
   if (useRattle()) {
      pos_prec *xold, *yold, *zold;
      darray::allocate(n, &xold, &yold, &zold);
      darray::copyin(g::q0, n, xold, x0.data());
      darray::copyin(g::q0, n, yold, y0.data());
      darray::copyin(g::q0, n, zold, z0.data());
      // SHAKE only uses the time step for velocity corrections, which it skips
      shake(1, xpos, ypos, zpos, xold, yold, zold);
      waitFor(g::q0);
      darray::deallocate(xold, yold, zold);
   }
   waitFor(g::q0);
   copyPosToXyz(true);

   ++s_ntry;
   s_retrying = true;
   s_failed = false;
   return true;
}
}
