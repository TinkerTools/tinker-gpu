#include "ff/energy.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "math/matexp.h"
#include "math/random.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

#include <cmath>

namespace tinker {
TINKER_FVOID2(acc1, cu1, kineticEnergy, energy_prec&, energy_prec (&)[3][3], int n, const double*, const vel_prec*,
   const vel_prec*, const vel_prec*);
void kineticEnergy(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n, const double* mass, const vel_prec* vx,
   const vel_prec* vy, const vel_prec* vz)
{
   TINKER_FCALL2(acc1, cu1, kineticEnergy, eksum_out, ekin_out, n, mass, vx, vy, vz);
}

void kineticExplicit(T_prec& temp_out, energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], const vel_prec* vx,
   const vel_prec* vy, const vel_prec* vz)
{
   kineticEnergy(eksum_out, ekin_out, n, mass, vx, vy, vz);
   temp_out = 2 * eksum_out / (mdstuf::nfree * units::gasconst);
}

void kinetic(T_prec& temp)
{
   kineticExplicit(temp, eksum, ekin, vx, vy, vz);
}
}

namespace tinker {
void bussiThermostat(time_prec dt_prec, T_prec temp_prec)
{
   double dt = dt_prec;
   double temp = temp_prec;

   double tautemp = bath::tautemp;
   double kelvin = bath::kelvin;
   int nfree = mdstuf::nfree;
   // double& eta = bath::eta;

   if (temp == 0)
      temp = 0.1;

   double c = std::exp(-dt / tautemp);
   double d = (1 - c) * (kelvin / temp) / nfree;
   double r = normal<double>();
   double s = chiSquared<double>(nfree - 1);
   double scale = c + (s + r * r) * d + 2 * r * std::sqrt(c * d);
   scale = std::sqrt(scale);
   if (r + std::sqrt(c / d) < 0)
      scale = -scale;
   // eta *= scale;

   vel_prec sc = scale;
   mdVelScale(sc, n, vx, vy, vz);
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         ekin[i][j] *= scale * scale;
      }
   }
}

static void invert3(double dst[3][3], const double src[3][3])
{
   double det = src[0][0] * (src[1][1] * src[2][2] - src[1][2] * src[2][1])
      - src[0][1] * (src[1][0] * src[2][2] - src[1][2] * src[2][0])
      + src[0][2] * (src[1][0] * src[2][1] - src[1][1] * src[2][0]);
   double invdet = 1 / det;

   dst[0][0] = (src[1][1] * src[2][2] - src[1][2] * src[2][1]) * invdet;
   dst[0][1] = (src[0][2] * src[2][1] - src[0][1] * src[2][2]) * invdet;
   dst[0][2] = (src[0][1] * src[1][2] - src[0][2] * src[1][1]) * invdet;
   dst[1][0] = (src[1][2] * src[2][0] - src[1][0] * src[2][2]) * invdet;
   dst[1][1] = (src[0][0] * src[2][2] - src[0][2] * src[2][0]) * invdet;
   dst[1][2] = (src[0][2] * src[1][0] - src[0][0] * src[1][2]) * invdet;
   dst[2][0] = (src[1][0] * src[2][1] - src[1][1] * src[2][0]) * invdet;
   dst[2][1] = (src[0][1] * src[2][0] - src[0][0] * src[2][1]) * invdet;
   dst[2][2] = (src[0][0] * src[1][1] - src[0][1] * src[1][0]) * invdet;
}

TINKER_FVOID2(acc1, cu1, monteCarloMolMove, double);
TINKER_FVOID2(acc1, cu1, monteCarloMolMoveAniso, const double (*)[3]);
void monteCarloBarostat(energy_prec epot, T_prec temp, bool semiiso, bool aniso)
{
   if (not bound::use_bounds)
      return;
   if (bath::isothermal)
      temp = bath::kelvin;

   FstrView volscale = bath::volscale;
   double third = 1.0 / 3.0;
   double volmove = bath::volmove;
   double kt = units::gasconst * temp;
   if (bath::isothermal)
      kt = units::gasconst * bath::kelvin;
   bool isotropic = true;
   if ((aniso || semiiso) and random<double>() > bath::isoratio)
      isotropic = false;

   bool semixy = true;
   if (semiiso and not isotropic and random<double>() > 0.5)
      semixy = false;

   // save the system state prior to trial box size change
   Box boxold;
   boxGetCurrent(boxold);
   double volold = boxVolume();
   double volnew = 0;
   double eold = epot;
   darray::copy(g::q0, n, x_pmonte, xpos);
   darray::copy(g::q0, n, y_pmonte, ypos);
   darray::copy(g::q0, n, z_pmonte, zpos);
   double step = volmove * (2 * random<double>() - 1);

   if (isotropic) {
      volnew = volold + step;
      double scale = std::pow(volnew / volold, third);

      lvec1 *= scale;
      lvec2 *= scale;
      lvec3 *= scale;
      boxSetCurrentRecip();

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMove, scale);
      }

      copyPosToXyz();

   } else if (semiiso) {
      double ascale[3][3] = {};
      ascale[0][0] = 1;
      ascale[1][1] = 1;
      ascale[2][2] = 1;

      volnew = volold + step;

      if (semixy) {
         double scale = std::sqrt(volnew / volold);
         lvec1 *= scale;
         lvec2 *= scale;
         ascale[0][0] = scale;
         ascale[1][1] = scale;
      } else {
         double scale = volnew / volold;
         lvec3 *= scale;
         ascale[2][2] = scale;
      }
      boxSetCurrentRecip();

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMoveAniso, ascale);
      }

      copyPosToXyz();

   } else if (aniso) {
      double rnd6 = 6 * random<double>();
      double voltarget = volold + step;
      bool offdiag = false;

      double ascale[3][3] = {};
      ascale[0][0] = 1;
      ascale[1][1] = 1;
      ascale[2][2] = 1;

      double scale = voltarget / volold;

      if (box_shape == BoxShape::MONO || box_shape == BoxShape::TRI) {
         if (rnd6 < 1) {
            ascale[0][0] = scale;
         } else if (rnd6 < 2) {
            ascale[1][1] = scale;
         } else if (rnd6 < 3) {
            ascale[2][2] = scale;
         } else if (rnd6 < 4) {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[0][1] = scale - 1;
            ascale[1][0] = scale - 1;
         } else if (rnd6 < 5) {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[0][2] = scale - 1;
            ascale[2][0] = scale - 1;
         } else {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[1][2] = scale - 1;
            ascale[2][1] = scale - 1;
         }

         double hbox0[3][3], hbox1[3][3];
         hbox0[0][0] = lvec1.x;
         hbox0[1][0] = 0;
         hbox0[2][0] = 0;
         hbox0[0][1] = lvec1.y;
         hbox0[1][1] = lvec2.y;
         hbox0[2][1] = 0;
         hbox0[0][2] = lvec1.z;
         hbox0[1][2] = lvec2.z;
         hbox0[2][2] = lvec3.z;
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               hbox1[i][j] = 0;
               for (int k = 0; k < 3; ++k) {
                  hbox1[i][j] += ascale[i][k] * hbox0[k][j];
               }
            }
         }
         double l0, l1, l2, a0 = 90, a1 = 90, a2 = 90;
         l0 = std::sqrt(hbox1[0][0] * hbox1[0][0] + hbox1[1][0] * hbox1[1][0] + hbox1[2][0] * hbox1[2][0]);
         l1 = std::sqrt(hbox1[0][1] * hbox1[0][1] + hbox1[1][1] * hbox1[1][1] + hbox1[2][1] * hbox1[2][1]);
         l2 = std::sqrt(hbox1[0][2] * hbox1[0][2] + hbox1[1][2] * hbox1[1][2] + hbox1[2][2] * hbox1[2][2]);
         if (box_shape == BoxShape::MONO or box_shape == BoxShape::TRI) {
            a1 = (hbox1[0][0] * hbox1[0][2] + hbox1[1][0] * hbox1[1][2] + hbox1[2][0] * hbox1[2][2]) / (l0 * l2);
            a1 = radian * std::acos(a1);
         }
         if (box_shape == BoxShape::TRI) {
            a0 = (hbox1[0][1] * hbox1[0][2] + hbox1[1][1] * hbox1[1][2] + hbox1[2][1] * hbox1[2][2]) / (l1 * l2);
            a0 = radian * std::acos(a0);
            a2 = (hbox1[0][0] * hbox1[0][1] + hbox1[1][0] * hbox1[1][1] + hbox1[2][0] * hbox1[2][1]) / (l0 * l1);
            a2 = radian * std::acos(a2);
         }
         Box newbox;
         boxLattice(newbox, box_shape, l0, l1, l2, a0, a1, a2);
         boxSetCurrent(newbox);

         if (offdiag) {
            double volmid = boxVolume();
            double scale2 = std::pow(voltarget / volmid, third);

            lvec1 *= scale2;
            lvec2 *= scale2;
            lvec3 *= scale2;
            boxSetCurrentRecip();

            hbox1[0][0] = lvec1.x;
            hbox1[1][0] = 0;
            hbox1[2][0] = 0;
            hbox1[0][1] = lvec1.y;
            hbox1[1][1] = lvec2.y;
            hbox1[2][1] = 0;
            hbox1[0][2] = lvec1.z;
            hbox1[1][2] = lvec2.z;
            hbox1[2][2] = lvec3.z;

            double h0inv[3][3];
            invert3(h0inv, hbox0);
            for (int i = 0; i < 3; ++i) {
               for (int j = 0; j < 3; ++j) {
                  ascale[i][j] = 0;
                  for (int k = 0; k < 3; ++k) {
                     ascale[i][j] += hbox1[i][k] * h0inv[k][j];
                  }
               }
            }
         }

         volnew = boxVolume();

      } else {
         if (rnd6 < 2) {
            ascale[0][0] = scale;
         } else if (rnd6 < 4) {
            ascale[1][1] = scale;
         } else {
            ascale[2][2] = scale;
         }

         lvec1 *= ascale[0][0];
         lvec2 *= ascale[1][1];
         lvec3 *= ascale[2][2];
         boxSetCurrentRecip();

         volnew = boxVolume();
      }

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMoveAniso, ascale);
      }

      copyPosToXyz();
   }

   // get the potential energy and PV work changes for trial move
   nblistRefresh();
   energy(calc::energy);
   energy_prec enew;
   copyEnergy(calc::energy, &enew);
   double dpot = enew - eold;
   double dpv = bath::atmsph * (volnew - volold) / units::prescon;

   // estimate the kinetic energy change as an ideal gas term
   double dkin = 0;
   if (volscale == "MOLECULAR") {
      dkin = molecule.nmol * kt * std::log(volold / volnew);
   }

   // acceptance ratio from Epot change, Ekin change and PV work
   double term = -(dpot + dpv + dkin) / kt;
   double expterm = std::exp(term);

   // reject the step, and restore values prior to trial change
   double exp_rdm = random<double>();
   if (exp_rdm > expterm) {
      esum = eold;
      boxSetCurrent(boxold);
      darray::copy(g::q0, n, xpos, x_pmonte);
      darray::copy(g::q0, n, ypos, y_pmonte);
      darray::copy(g::q0, n, zpos, z_pmonte);
      copyPosToXyz();
      nblistRefresh();
   }
}

TINKER_FVOID2(acc1, cu1, scaleBarostatAtomMove, double, double, double);
TINKER_FVOID2(acc1, cu1, scaleBarostatAtomMoveAniso, const double (*)[3]);
TINKER_FVOID2(acc1, cu1, scaleVelocity, double, double, double);
TINKER_FVOID2(acc1, cu1, scaleVelocityAniso, const double (*)[3]);

void scaleBarostat(time_prec dt, bool semiiso, bool aniso, ScaleBaroEnum be)
{
   if (not bound::use_bounds)
      return;
   bool use_berendsen = (be == ScaleBaroEnum::BERENDSEN);
   bool use_bussi = (be == ScaleBaroEnum::BUSSI);
   bool isotropic = true;
   if (aniso || semiiso)
      isotropic = false;
   double vol = boxVolume();
   double factor = units::prescon / vol;
   double stress[3][3];
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         int iv = 3 * i + j;
         stress[i][j] = factor * (2 * ekin[i][j] - vir[iv]);
      }
   }
   const double third = 1.0 / 3;
   double pres = (stress[0][0] + stress[1][1] + stress[2][2]) * third;

   if (isotropic) {
      double scale;
      if (use_berendsen) {
         double eps = (bath::compress * dt / bath::taupres) * (pres - bath::atmsph);
         scale = std::exp(third * eps);
      } else if (use_bussi) {
         double kt = units::gasconst * bath::kelvin;
         double betat = units::prescon * bath::compress;
         double dw = normal<double>();
         double eps = (bath::compress * dt / bath::taupres) * (pres - bath::atmsph);
         double deps = std::sqrt(2 * kt * betat * dt / (vol * bath::taupres));
         scale = std::exp(third * (eps + deps * dw));
      }

      lvec1 *= scale;
      lvec2 *= scale;
      lvec3 *= scale;
      boxSetCurrentRecip();

      TINKER_FCALL2(acc1, cu1, scaleBarostatAtomMove, scale, scale, scale);
      if (use_bussi)
         TINKER_FCALL2(acc1, cu1, scaleVelocity, 1 / scale, 1 / scale, 1 / scale);
      copyPosToXyz();
   } else if (semiiso) {
      double tension = 0.0;
      double scalexy;
      double scalez;
      if (use_berendsen) {
         double eps = third * (bath::compress * dt / bath::taupres);
         double term = 0.5 * (stress[0][0] + stress[1][1]) + (tension / lvec3.z) - bath::atmsph;
         scalexy = std::exp(eps * term);
         term = stress[2][2] - bath::atmsph;
         scalez = std::exp(eps * term);
      } else if (use_bussi) {
         double kt = units::gasconst * bath::kelvin;
         double betat = units::prescon * bath::compress;
         double eps = bath::compress * dt / bath::taupres;
         double depsxy = std::sqrt(kt * betat * dt / (3 * vol * bath::taupres));
         double depsz = std::sqrt(2 * kt * betat * dt / (3 * vol * bath::taupres));
         double dw = normal<double>();
         double term = 0.5 * (stress[0][0] + stress[1][1]) + (tension / lvec3.z) - bath::atmsph;
         scalexy = std::exp(third * eps * term + depsxy * dw);
         dw = normal<double>();
         term = stress[2][2] - bath::atmsph;
         scalez = std::exp(third * eps * term + depsz * dw);
      }

      lvec1 *= scalexy;
      lvec2 *= scalexy;
      lvec3 *= scalez;
      boxSetCurrentRecip();

      TINKER_FCALL2(acc1, cu1, scaleBarostatAtomMove, scalexy, scalexy, scalez);
      if (use_bussi)
         TINKER_FCALL2(acc1, cu1, scaleVelocity, 1 / scalexy, 1 / scalexy, 1 / scalez);
      copyPosToXyz();
   } else if (aniso) {
      double ascale[3][3];
      if (use_berendsen) {
         double eps = third * (bath::compress * dt / bath::taupres);
         double amat[3][3];
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               if (j == i) {
                  amat[i][j] = stress[i][j] - bath::atmsph;
               } else {
                  amat[i][j] = stress[i][j];
               }
            }
         }
         matExp3(ascale, amat, eps);
      } else if (use_bussi) {
         double kt = units::gasconst * bath::kelvin;
         double betat = units::prescon * bath::compress;
         double eps = third * (bath::compress * dt / bath::taupres);
         double deps = std::sqrt((2 * third) * kt * betat * dt / (vol * bath::taupres));
         double amat[3][3];
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               double dw = normal<double>();
               if (j == i) {
                  double term = stress[i][j] - bath::atmsph;
                  term += units::prescon * kt / vol;
                  amat[i][j] = eps * term + deps * dw;
               } else {
                  amat[i][j] = eps * stress[i][j] + deps * dw;
               }
            }
         }
         matExp3(ascale, amat, 1.0);
      }

      double hbox0[3][3], hbox1[3][3];
      hbox0[0][0] = lvec1.x;
      hbox0[1][0] = 0;
      hbox0[2][0] = 0;
      hbox0[0][1] = lvec1.y;
      hbox0[1][1] = lvec2.y;
      hbox0[2][1] = 0;
      hbox0[0][2] = lvec1.z;
      hbox0[1][2] = lvec2.z;
      hbox0[2][2] = lvec3.z;
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            hbox1[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
               hbox1[i][j] += ascale[i][k] * hbox0[k][j];
            }
         }
      }
      double l0, l1, l2, a0 = 90, a1 = 90, a2 = 90;
      l0 = std::sqrt(hbox1[0][0] * hbox1[0][0] + hbox1[1][0] * hbox1[1][0] + hbox1[2][0] * hbox1[2][0]);
      l1 = std::sqrt(hbox1[0][1] * hbox1[0][1] + hbox1[1][1] * hbox1[1][1] + hbox1[2][1] * hbox1[2][1]);
      l2 = std::sqrt(hbox1[0][2] * hbox1[0][2] + hbox1[1][2] * hbox1[1][2] + hbox1[2][2] * hbox1[2][2]);
      if (box_shape == BoxShape::MONO or box_shape == BoxShape::TRI) {
         a1 = (hbox1[0][0] * hbox1[0][2] + hbox1[1][0] * hbox1[1][2] + hbox1[2][0] * hbox1[2][2]) / (l0 * l2);
         a1 = radian * std::acos(a1);
      }
      if (box_shape == BoxShape::TRI) {
         a0 = (hbox1[0][1] * hbox1[0][2] + hbox1[1][1] * hbox1[1][2] + hbox1[2][1] * hbox1[2][2]) / (l1 * l2);
         a0 = radian * std::acos(a0);
         a2 = (hbox1[0][0] * hbox1[0][1] + hbox1[1][0] * hbox1[1][1] + hbox1[2][0] * hbox1[2][1]) / (l0 * l1);
         a2 = radian * std::acos(a2);
      }
      Box newbox;
      boxLattice(newbox, box_shape, l0, l1, l2, a0, a1, a2);
      boxSetCurrent(newbox);

      hbox1[0][0] = lvec1.x;
      hbox1[1][0] = 0;
      hbox1[2][0] = 0;
      hbox1[0][1] = lvec1.y;
      hbox1[1][1] = lvec2.y;
      hbox1[2][1] = 0;
      hbox1[0][2] = lvec1.z;
      hbox1[1][2] = lvec2.z;
      hbox1[2][2] = lvec3.z;

      double h0inv[3][3];
      invert3(h0inv, hbox0);
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            ascale[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
               ascale[i][j] += hbox1[i][k] * h0inv[k][j];
            }
         }
      }

      TINKER_FCALL2(acc1, cu1, scaleBarostatAtomMoveAniso, ascale);
      if (use_bussi) {
         double ainv[3][3];
         invert3(ainv, ascale);
         double ainvT[3][3];
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               ainvT[i][j] = ainv[j][i];
         TINKER_FCALL2(acc1, cu1, scaleVelocityAniso, ainvT);
      }
      copyPosToXyz();
   }
}
}
