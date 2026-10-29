#pragma once
#include "math/random.h"
#include <algorithm>
#include <cmath>
#include <vector>

/// \file
/// The estimators of "bar" mode 2, FEP and BAR free energies and enthalpies of
/// one pair of states, with the bootstrap versions that use the exponentials of
/// each frame found once (tinkerDev's barpair.f).
namespace tinker {
namespace bar {
/// Potential energies and volumes of the frames selected from both trajectories.
struct BarData
{
   std::vector<double> ua0, ua1; ///< Energies of trajectory A frames in states 0 and 1.
   std::vector<double> ub0, ub1; ///< Energies of trajectory B frames in states 0 and 1.
   std::vector<double> vola;     ///< Volumes of trajectory A frames, or 0 if not used.
   std::vector<double> volb;     ///< Volumes of trajectory B frames, or 0 if not used.
};

/// Fills idx with n frame indices drawn with replacement.
inline void barDraw(int n, std::vector<int>& idx)
{
   idx.resize(n);
   for (int i = 0; i < n; ++i)
      idx[i] = std::min((int)(n * random<double>()), n - 1);
}

/// Finds the free energy and enthalpy change from the sampled state to the
/// other state via thermodynamic perturbation over a list of frames, which may
/// repeat, along with the average energy and volume of those frames. The
/// exponentials are shifted by their largest exponent so the sum cannot
/// overflow, and the enthalpy includes the PV term.
inline void barFep(const std::vector<int>& idx, const std::vector<double>& uo, const std::vector<double>& ut,
   const std::vector<double>& vol, double rt, double pres, double& dg, double& dh, double& uave, double& vave)
{
   const int n = idx.size();
   double xmax = 0;
   for (int i = 0; i < n; ++i) {
      int j = idx[i];
      double x = (uo[j] - ut[j]) / rt;
      if (i == 0 or x > xmax)
         xmax = x;
   }

   double wsum = 0, hwsum = 0, usum = 0, vsum = 0;
   for (int i = 0; i < n; ++i) {
      int j = idx[i];
      double w = std::exp((uo[j] - ut[j]) / rt - xmax);
      wsum += w;
      hwsum += (ut[j] + pres * vol[j]) * w;
      usum += uo[j];
      vsum += vol[j];
   }
   uave = usum / n;
   vave = vsum / n;
   dg = -rt * (xmax + std::log(wsum / n));
   dh = hwsum / wsum - (uave + pres * vave);
}

/// Solves the Bennett acceptance ratio equation for the free energy change from
/// state 0 to state 1 by iteration over lists of frames sampled in each state,
/// starting from cinit; also finds the analytical error of Bennett. Records the
/// value at each iteration in hist if it is not null, and returns false if the
/// iteration does not converge. The Fermi functions include the optimal shift of
/// Bennett, ln(nb/na), which gives the lowest variance when the numbers of frames
/// differ, and the equation then balances the sums of the Fermi functions over
/// the frames of each state.
inline bool barIter(const std::vector<int>& ia, const std::vector<int>& ib, const BarData& d, double rta, double rtb,
   double cinit, int maxiter, double& cnew, double& stdev, std::vector<double>* hist)
{
   const double eps = 0.0001;
   const double rt = 0.5 * (rta + rtb);
   const int na = ia.size(), nb = ib.size();
   const double frma = na, frmb = nb;
   const double lnr = std::log(frmb / frma);

   cnew = cinit;
   stdev = 0;
   if (hist)
      hist->clear();
   for (int iter = 0;; ++iter) {
      double cold = cnew;
      double top = 0, top2 = 0;
      for (int i = 0; i < nb; ++i) {
         int j = ib[i];
         double fterm = 1.0 / (1.0 + std::exp((d.ub0[j] - d.ub1[j] + cold) / rtb + lnr));
         top += fterm;
         top2 += fterm * fterm;
      }
      double bot = 0, bot2 = 0;
      for (int i = 0; i < na; ++i) {
         int j = ia[i];
         double fterm = 1.0 / (1.0 + std::exp((d.ua1[j] - d.ua0[j] - cold) / rta - lnr));
         bot += fterm;
         bot2 += fterm * fterm;
      }
      cnew = rt * std::log(top / bot) + cold;
      stdev = rt * std::sqrt(std::max(0.0, (bot2 - bot * bot / frma) / (bot * bot) + (top2 - top * top / frmb) / (top * top)));
      if (hist)
         hist->push_back(cnew);
      if (std::fabs(cnew - cold) < eps)
         return true;
      else if (iter >= maxiter)
         return false;
      else if (not(std::fabs(cnew) < 1.0e100))
         return false;
   }
}

/// Finds the enthalpy change from state 0 to state 1 via the BAR estimator of
/// Wyczalkowski et al. over lists of frames sampled in each state, given the BAR
/// free energy change cbar for the same frames. The state energies include the
/// PV term, which cancels in the energy differences, and the Fermi functions
/// include the shift of Bennett used by barIter, so the terms are sums over the
/// frames of both states.
inline double barHbar(const std::vector<int>& ia, const std::vector<int>& ib, const BarData& d, double rta, double rtb,
   double pres, double cbar)
{
   const double lnr = std::log((double)ib.size() / ia.size());

   // find the terms from the state 0 frames
   double fsum = 0, fvsum = 0, fbvsum = 0, vsum = 0, fbsum0 = 0;
   for (int j : ia) {
      double fore = 1.0 / (1.0 + std::exp((d.ua1[j] - d.ua0[j] - cbar) / rta - lnr));
      double back = 1.0 / (1.0 + std::exp((d.ua0[j] - d.ua1[j] + cbar) / rta + lnr));
      double h = d.ua0[j] + pres * d.vola[j];
      fsum += fore;
      fvsum += fore * h;
      fbvsum += fore * back * (d.ua1[j] - d.ua0[j]);
      vsum += h;
      fbsum0 += fore * back;
   }
   double alpha0 = fvsum - fsum * (vsum / ia.size()) + fbvsum;

   // find the terms from the state 1 frames
   double bsum = 0, bvsum = 0, fbsum1 = 0;
   fbvsum = 0, vsum = 0;
   for (int j : ib) {
      double fore = 1.0 / (1.0 + std::exp((d.ub1[j] - d.ub0[j] - cbar) / rtb - lnr));
      double back = 1.0 / (1.0 + std::exp((d.ub0[j] - d.ub1[j] + cbar) / rtb + lnr));
      double h = d.ub1[j] + pres * d.volb[j];
      bsum += back;
      bvsum += back * h;
      fbvsum += fore * back * (d.ub1[j] - d.ub0[j]);
      vsum += h;
      fbsum1 += fore * back;
   }
   double alpha1 = bvsum - bsum * (vsum / ib.size()) - fbvsum;
   return (alpha0 - alpha1) / (fbsum0 + fbsum1);
}

/// The exponentials of each frame used by the bootstrap versions of barFep,
/// barIter and barHbar, found once so that a bootstrap trial needs none per
/// frame (tinkerDev's bootprep). The FEP weights are shifted by their largest
/// exponent over all of the frames, and the Fermi exponentials are taken at a
/// reference free energy near the values found in the trials, without the shift
/// of Bennett, which depends on the numbers of frames in a trial.
struct BootTerms
{
   double cref;          ///< Reference free energy change of the Fermi exponentials.
   double xmaxa, xmaxb;  ///< Largest forward and backward FEP exponents.
   std::vector<double> wa, hwa; ///< Shifted forward FEP weights, and state 1 enthalpies times them.
   std::vector<double> wb, hwb; ///< Shifted backward FEP weights, and state 0 enthalpies times them.
   std::vector<double> ea, eb;  ///< Exponentials of the state 0 and state 1 Fermi function arguments.

   BootTerms(const BarData& d, double rta, double rtb, double pres, double cref)
      : cref(cref)
   {
      const int na = d.ua0.size(), nb = d.ub0.size();
      xmaxa = (d.ua0[0] - d.ua1[0]) / rta;
      for (int i = 1; i < na; ++i)
         xmaxa = std::max(xmaxa, (d.ua0[i] - d.ua1[i]) / rta);
      xmaxb = (d.ub1[0] - d.ub0[0]) / rtb;
      for (int i = 1; i < nb; ++i)
         xmaxb = std::max(xmaxb, (d.ub1[i] - d.ub0[i]) / rtb);
      for (int i = 0; i < na; ++i) {
         wa.push_back(std::exp((d.ua0[i] - d.ua1[i]) / rta - xmaxa));
         hwa.push_back((d.ua1[i] + pres * d.vola[i]) * wa[i]);
         ea.push_back(std::exp((d.ua1[i] - d.ua0[i] - cref) / rta));
      }
      for (int i = 0; i < nb; ++i) {
         wb.push_back(std::exp((d.ub1[i] - d.ub0[i]) / rtb - xmaxb));
         hwb.push_back((d.ub0[i] + pres * d.volb[i]) * wb[i]);
         eb.push_back(std::exp((d.ub0[i] - d.ub1[i] + cref) / rtb));
      }
   }
};

/// Gives the same values as barFep from the weights w and hw shifted by xmax,
/// and calls barFep if every weight of the selected frames underflows.
inline void bootFep(const std::vector<int>& idx, const std::vector<double>& uo, const std::vector<double>& ut,
   const std::vector<double>& vol, const std::vector<double>& w, const std::vector<double>& hw, double xmax,
   double rt, double pres, double& dg, double& dh, double& uave, double& vave)
{
   double wsum = 0, hwsum = 0, usum = 0, vsum = 0;
   for (int j : idx) {
      wsum += w[j];
      hwsum += hw[j];
      usum += uo[j];
      vsum += vol[j];
   }
   if (wsum == 0) {
      barFep(idx, uo, ut, vol, rt, pres, dg, dh, uave, vave);
      return;
   }
   const int n = idx.size();
   uave = usum / n;
   vave = vsum / n;
   dg = -rt * (xmax + std::log(wsum / n));
   dh = hwsum / wsum - (uave + pres * vave);
}

/// Solves the same equation as barIter from the Fermi exponentials of t,
/// without the analytical error, and returns false if it does not converge.
inline bool bootIter(const std::vector<int>& ia, const std::vector<int>& ib, const BootTerms& t, double rta, double rtb,
   double cinit, int maxiter, double& cnew)
{
   const double eps = 0.0001;
   const double rt = 0.5 * (rta + rtb);
   const double lnr = std::log((double)ib.size() / ia.size());
   cnew = cinit;
   for (int iter = 0;; ++iter) {
      double cold = cnew;
      double ka = std::exp(-(cold - t.cref) / rta - lnr), kb = std::exp((cold - t.cref) / rtb + lnr);
      double top = 0, bot = 0;
      for (int j : ib)
         top += 1.0 / (1.0 + t.eb[j] * kb);
      for (int j : ia)
         bot += 1.0 / (1.0 + t.ea[j] * ka);
      cnew = rt * std::log(top / bot) + cold;
      if (std::fabs(cnew - cold) < eps)
         return true;
      else if (iter >= maxiter)
         return false;
      else if (not(std::fabs(cnew) < 1.0e100))
         return false;
   }
}

/// Gives the same value as barHbar from the Fermi exponentials of t. Each Fermi
/// function and its complement come from one exponential, and stay finite when
/// it is zero or infinite.
inline double bootHbar(const std::vector<int>& ia, const std::vector<int>& ib, const BarData& d, const BootTerms& t,
   double rta, double rtb, double pres, double cbar)
{
   const double lnr = std::log((double)ib.size() / ia.size());
   const double ka = std::exp(-(cbar - t.cref) / rta - lnr), kb = std::exp((cbar - t.cref) / rtb + lnr);

   // find the terms from the state 0 frames
   double fsum = 0, fvsum = 0, fbvsum = 0, vsum = 0, fbsum0 = 0;
   for (int j : ia) {
      double x = t.ea[j] * ka;
      double fore = 1.0 / (1.0 + x), back = 1.0 / (1.0 + 1.0 / x);
      double h = d.ua0[j] + pres * d.vola[j];
      fsum += fore;
      fvsum += fore * h;
      fbvsum += fore * back * (d.ua1[j] - d.ua0[j]);
      vsum += h;
      fbsum0 += fore * back;
   }
   double alpha0 = fvsum - fsum * (vsum / ia.size()) + fbvsum;

   // find the terms from the state 1 frames
   double bsum = 0, bvsum = 0, fbsum1 = 0;
   fbvsum = 0, vsum = 0;
   for (int j : ib) {
      double x = t.eb[j] * kb;
      double back = 1.0 / (1.0 + x), fore = 1.0 / (1.0 + 1.0 / x);
      double h = d.ub1[j] + pres * d.volb[j];
      bsum += back;
      bvsum += back * h;
      fbvsum += fore * back * (d.ub1[j] - d.ub0[j]);
      vsum += h;
      fbsum1 += fore * back;
   }
   double alpha1 = bvsum - bsum * (vsum / ib.size()) - fbvsum;
   return (alpha0 - alpha1) / (fbsum0 + fbsum1);
}

/// The default number of bootstrap trials for n frame evaluations per trial,
/// about 1e8 evaluations in all, between 2000 trials, which find errors to about
/// 1.5 percent, and 10000 trials (tinkerDev's barnbst).
inline int barTrials(int n)
{
   return std::max(2000, std::min(10000, (int)std::round(1.e8 / std::max(1, n))));
}
}
}
