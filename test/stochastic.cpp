#include "ff/atom.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/stochastic.h"
#include "seq/philox.h"
#include "tool/accasync.h"
#include "tool/darray.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <tinker/detail/bath.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/stodyn.hh>
#include <tinker/detail/units.hh>

#include "test.h"
#include "testrt.h"

using namespace tinker;

namespace {
// Drives sdSetTimeStep() through the Fortran-side friction, the only input it
// reads besides the time step.
struct Coef
{
   double pfric, vfric, afric, pterm, vterm, rho;

   Coef(double gamma, double dt)
   {
      double save = stodyn::friction;
      stodyn::friction = gamma;
      sdSetTimeStep(dt);
      sdGetCoefficients(&pfric, &vfric, &afric, &pterm, &vterm, &rho);
      stodyn::friction = save;
   }
};

double relerr(double a, double b)
{
   double s = std::fabs(b);
   return s > 0 ? std::fabs(a - b) / s : std::fabs(a - b);
}
}

TEST_CASE("SD-Coefficients", "[md][stochastic]")
{
   const double dt = 0.001; // ps

   SECTION("SeriesBranchContinuity")
   {
      // sdSetTimeStep switches from the closed forms to the series expansions at
      // gdt = 0.05. Both sides must agree across that seam; a gross typo in a
      // series term shows up here and essentially nowhere else cheaply. The
      // tolerance has to leave room for the two branches being sampled at
      // slightly different gdt, which is what sets the 1e-9 floor.
      const double gdt = 0.05, eps = 1e-12;
      Coef hi(gdt * (1 + eps) / dt, dt); // closed form
      Coef lo(gdt * (1 - eps) / dt, dt); // series

      REQUIRE(relerr(lo.pfric, hi.pfric) < 1e-9);
      REQUIRE(relerr(lo.vfric, hi.vfric) < 1e-9);
      REQUIRE(relerr(lo.afric, hi.afric) < 1e-9);
      REQUIRE(relerr(lo.pterm, hi.pterm) < 1e-9);
      REQUIRE(relerr(lo.vterm, hi.vterm) < 1e-9);
      REQUIRE(relerr(lo.rho, hi.rho) < 1e-9);
   }

   SECTION("HighPrecisionReferences")
   {
      // Reference values computed in 50-digit arithmetic for dt = 0.001 ps.
      // This is what actually pins down every term of every series: the sweep
      // spans the series range, so early terms dominate at the small end and
      // late terms at the large end.
      //
      // The tolerance is looser near gdt = 0.05 because that is where both
      // branches are at their worst, which is precisely why the threshold sits
      // there. Just below it the truncated series carries its largest error
      // (2.9e-13 in pterm at gdt = 0.049, from the omitted g**10 term); just
      // above it the closed form carries its largest error (6.3e-13 in pterm at
      // gdt = 0.05, since pterm is an O(gdt**3) quantity assembled from O(1)
      // terms). Away from the seam both are at machine epsilon. Going the other
      // way, the closed form degrades fast: by gdt = 0.005 it is wrong in its
      // 9th digit and by gdt = 0.0005 in its 6th.
      struct
      {
         double gdt, pfric, vfric, afric, pterm, vterm, rho, tol;
      } ref[] = {
         // series branch
         {0.0001, 9.9990000499983e-01, 9.9995000166663e-04, 4.9998333374999e-07, //
            6.6661666899992e-13, 1.9998000133327e-04, 8.6601457823686e-01, 1e-13},
         {0.001, 9.9900049983338e-01, 9.9950016662501e-04, 4.9983337499167e-07, //
            6.6616689991669e-10, 1.9980013326669e-03, 8.6591712760996e-01, 1e-13},
         {0.005, 9.9501247919268e-01, 9.9750416146354e-04, 4.9916770729253e-07, //
            8.3021561199836e-08, 9.9501662508319e-03, 8.6548356341242e-01, 1e-13},
         {0.02, 9.8019867330676e-01, 9.9006633466223e-04, 4.9668326688826e-07, //
            5.2540746979994e-06, 3.9210560847677e-02, 8.6385117742354e-01, 1e-13},
         {0.049, 9.5218112969850e-01, 9.7589531227541e-04, 4.9193240254263e-07, //
            7.5615040098492e-05, 9.3351096246079e-02, 8.6066634167549e-01, 1e-12},
         // closed-form branch
         {0.05, 9.5122942450071e-01, 9.7541150998572e-04, 4.9176980028560e-07, //
            8.0279966896463e-05, 9.5162581964040e-02, 8.6055584734270e-01, 1e-11},
         {0.5, 6.0653065971263e-01, 7.8693868057473e-04, 4.2612263885053e-07, //
            5.8243197679091e-02, 6.3212055882856e-01, 8.0686194215607e-01, 1e-13},
         {5.0, 6.7379469990855e-03, 1.9865241060018e-04, 1.6026951787996e-07, //
            7.0269063880666e+00, 9.9995460007024e-01, 3.7218208315845e-01, 1e-13},
      };
      for (auto& r : ref) {
         Coef c(r.gdt / dt, dt);
         CAPTURE(r.gdt);
         REQUIRE(relerr(c.pfric, r.pfric) < r.tol);
         REQUIRE(relerr(c.vfric, r.vfric) < r.tol);
         REQUIRE(relerr(c.afric, r.afric) < r.tol);
         REQUIRE(relerr(c.pterm, r.pterm) < r.tol);
         REQUIRE(relerr(c.vterm, r.vterm) < r.tol);
         REQUIRE(relerr(c.rho, r.rho) < r.tol);
      }
   }

   SECTION("ZeroFrictionLimit")
   {
      Coef c(0.0, dt);
      REQUIRE(c.pfric == Approx(1.0).margin(0));
      REQUIRE(c.vfric == Approx(dt).margin(0));
      REQUIRE(c.afric == Approx(0.5 * dt * dt).margin(0));
      REQUIRE(c.pterm == 0.0);
      REQUIRE(c.vterm == 0.0);

      // and the series branch has to approach that limit continuously
      Coef tiny(1e-8 / dt, dt);
      REQUIRE(relerr(tiny.pfric, 1.0) < 1e-7);
      REQUIRE(relerr(tiny.vfric, dt) < 1e-7);
      REQUIRE(relerr(tiny.afric, 0.5 * dt * dt) < 1e-7);
   }

   SECTION("RhoIsAValidCorrelation")
   {
      for (double gdt : {1e-6, 1e-4, 1e-2, 0.05, 1.0, 10.0, 100.0}) {
         Coef c(gdt / dt, dt);
         REQUIRE(c.rho > 0.0);
         REQUIRE(c.rho < 1.0);
      }
      // the free-particle limit of the position/velocity noise correlation
      Coef c(1e-6 / dt, dt);
      REQUIRE(c.rho == Approx(std::sqrt(3.0) / 2).epsilon(1e-6));
   }
}

TEST_CASE("SD-Philox", "[md][stochastic]")
{
   SECTION("KnownAnswerTest")
   {
      // Published Random123 test vectors for Philox4x32-10.
      struct
      {
         uint32_t ctr[4], key[2], out[4];
      } kat[] = {
         {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
         {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
            {0xffffffff, 0xffffffff},
            {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
         {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
            {0xa4093822, 0x299f31d0},
            {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
      };
      for (auto& t : kat) {
         uint32_t c[4] = {t.ctr[0], t.ctr[1], t.ctr[2], t.ctr[3]};
         philox4x32(c, t.key[0], t.key[1]);
         for (int i = 0; i < 4; ++i)
            REQUIRE(c[i] == t.out[i]);
      }
   }

   SECTION("Reproducible")
   {
      double a0, a1, b0, b1;
      philoxNormal2(a0, a1, 7u, 3u, 11u, 1u);
      philoxNormal2(b0, b1, 7u, 3u, 11u, 1u);
      REQUIRE(a0 == b0);
      REQUIRE(a1 == b1);

      // a different atom, step or slot must give different numbers
      philoxNormal2(b0, b1, 7u, 3u, 12u, 1u);
      REQUIRE(a0 != b0);
      philoxNormal2(b0, b1, 7u, 4u, 11u, 1u);
      REQUIRE(a0 != b0);
      philoxNormal2(b0, b1, 7u, 3u, 11u, 2u);
      REQUIRE(a0 != b0);
   }

   SECTION("MomentsAndIndependence")
   {
      // The pair returned by philoxNormal2 has to be two *independent* standard
      // normals; the correlated pair that stochastic dynamics actually needs is
      // built from them in the kernel, and is checked end to end by
      // the Momentum section of the SD-Pbc-* and SD-NoPbc-* cases below.
      const int nsample = 1000000;
      double s0 = 0, s1 = 0, s00 = 0, s11 = 0, s01 = 0;
      for (int i = 0; i < nsample; ++i) {
         double z0, z1;
         philoxNormal2(z0, z1, 42u, 1u, i, 0u);
         s0 += z0; s1 += z1; s00 += z0 * z0; s11 += z1 * z1; s01 += z0 * z1;
      }
      double m0 = s0 / nsample, m1 = s1 / nsample;
      double v0 = s00 / nsample - m0 * m0, v1 = s11 / nsample - m1 * m1;
      double cov = s01 / nsample - m0 * m1;

      const double tol = 4.0 / std::sqrt((double)nsample);
      REQUIRE(std::fabs(m0) < tol);
      REQUIRE(std::fabs(m1) < tol);
      REQUIRE(std::fabs(v0 - 1) < 4 * std::sqrt(2.0 / nsample));
      REQUIRE(std::fabs(v1 - 1) < 4 * std::sqrt(2.0 / nsample));
      REQUIRE(std::fabs(cov / std::sqrt(v0 * v1)) < tol);
   }
}

namespace {
// Each dynamics test below runs in four cells: with and without a periodic
// box, and with and without REMOVE-INERTIA. All four use the same 20-atom
// TEMOA host-guest system with AMOEBA mutual polarization.
//
// The periodic cell puts the molecule in a 30 Angstrom box with 6 Angstrom
// cutoffs and no Ewald. With the default 9 Angstrom cutoffs every pair in the
// molecule would be inside the cutoff and the box would change nothing; 6
// Angstrom cuts real pairs, so the image and cutoff code paths change the
// numbers. Ewald is avoided because its grid spread is not bit-reproducible.
//
// The restart file carries the box, so the periodic cell needs its own,
// g3pbc.dyn. It is g3.dyn with the box set and the current accelerations
// replaced by the forces under the periodic cell's keys, which Fortran
// testgrad.x gives at the restart positions: Fortran takes the first step's
// accelerations from the restart file, while Tinker9 computes them, and the
// two only agree if the file is consistent with the keys.
struct Cell
{
   bool pbc, rest;
};

const int cell_irest = 3; // REMOVE-INERTIA period in the removal cells

std::string cellKey(Cell c)
{
   std::string k;
   if (c.pbc)
      k += "\nA-AXIS 30.0\nVDW-CUTOFF 6.0\nMPOLE-CUTOFF 6.0\n";
   if (c.rest)
      k += "\nREMOVE-INERTIA " + std::to_string(cell_irest) + "\n";
   return k;
}

bool removedAt(Cell c, int istep)
{
   return c.rest and istep % cell_irest == 0;
}

struct G3Setup
{
   int natom, dorest, irest, nfree;
};

// One dynamics run on G3. Every step is followed by mdrest(), as it is in
// mdIntegrateData(), and then by perStep(i); perStep(0) sees the starting
// state. esum and eksum are not touched by mdrest(), so perStep() still sees
// the values of the step itself.
template <class MakeIntegrator, class PerStep>
G3Setup runG3(Cell c, std::string keyExtra, double kelvin, int nsteps, double dt, //
   MakeIntegrator makeIntegrator, PerStep perStep)
{
   const char* k = "test_g3.key";
   const char* d = "test_g3.dyn";
   const char* x = "test_g3.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/stochastic/g3.key", k, cellKey(c) + keyExtra);
   TestFile fd(c.pbc ? TINKER9_DIRSTR "/test/file/stochastic/g3pbc.dyn" //
                     : TINKER9_DIRSTR "/test/file/stochastic/g3.dyn",
      d);
   TestFile fx(TINKER9_DIRSTR "/test/file/stochastic/g3.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/stochastic/hostsG3.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   TestSession session(argc, argv);
   testMdInit(kelvin, 0.);

   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md;
   session.init();

   G3Setup s{n, mdstuf::dorest, mdstuf::irest, mdstuf::nfree};
   int old = inform::iwrite;
   inform::iwrite = 1;
   {
      auto* intg = makeIntegrator();
      perStep(0);
      for (int i = 1; i <= nsteps; ++i) {
         intg->dynamic(i, dt);
         mdrest(i);
         perStep(i);
      }
      delete intg;
   }
   inform::iwrite = old;

   session.end();
   TestRemoveFileOnExit arc("test_g3.arc");
   bath::kelvin = 0.;
   bath::isothermal = 0;
   return s;
}

// Overall linear momentum p and angular momentum l about the center of mass,
// each with a scale to judge "zero" against: sum m|v| for p, and
// sum m|r-rcom||v-vcom| for l.
struct Momenta
{
   double p[3], l[3], pscale, lscale;
};

double norm3(const double* a)
{
   return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

Momenta measureMomenta()
{
   const int natom = n;
   std::vector<double> m(natom);
   std::vector<pos_prec> x(natom), y(natom), z(natom);
   std::vector<vel_prec> u(natom), v(natom), w(natom);
   darray::copyout(g::q0, natom, m.data(), mass);
   darray::copyout(g::q0, natom, x.data(), xpos);
   darray::copyout(g::q0, natom, y.data(), ypos);
   darray::copyout(g::q0, natom, z.data(), zpos);
   darray::copyout(g::q0, natom, u.data(), vx);
   darray::copyout(g::q0, natom, v.data(), vy);
   darray::copyout(g::q0, natom, w.data(), vz);
   waitFor(g::q0);

   Momenta a{};
   double mtot = 0, r[3] = {0};
   for (int i = 0; i < natom; ++i) {
      mtot += m[i];
      r[0] += m[i] * x[i], r[1] += m[i] * y[i], r[2] += m[i] * z[i];
      a.p[0] += m[i] * u[i], a.p[1] += m[i] * v[i], a.p[2] += m[i] * w[i];
      a.pscale += m[i] * std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
   }
   for (int i = 0; i < natom; ++i) {
      double dx = x[i] - r[0] / mtot, dy = y[i] - r[1] / mtot, dz = z[i] - r[2] / mtot;
      double du = u[i] - a.p[0] / mtot, dv = v[i] - a.p[1] / mtot, dw = w[i] - a.p[2] / mtot;
      a.l[0] += m[i] * (dy * dw - dz * dv);
      a.l[1] += m[i] * (dz * du - dx * dw);
      a.l[2] += m[i] * (dx * dv - dy * du);
      a.lscale += m[i] * std::sqrt(dx * dx + dy * dy + dz * dz) * std::sqrt(du * du + dv * dv + dw * dw);
   }
   return a;
}

// Advances the linear momentum by one step without removal. Summed over the
// atoms, the force terms of sdPos() and sdVel2() cancel, since the internal
// forces sum to zero, so p' = pfric p + sum_i m_i vrand_i. The noise is
// regenerated here from the same Philox stream the kernel draws from, which
// makes this an independent prediction of p at every step.
void advanceMomentum(double (&p)[3], int istep, unsigned int seed, const std::vector<double>& m)
{
   double pfric, vterm, rho;
   sdGetCoefficients(&pfric, nullptr, nullptr, nullptr, &vterm, &rho);
   double vsig = std::sqrt(units::boltzmann * bath::kelvin * vterm);
   double rhoc = std::sqrt(1 - rho * rho);
   double dp[3] = {0};
   for (int i = 0; i < (int)m.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
         double pn, vn;
         philoxNormal2(pn, vn, seed, istep, i, j);
         dp[j] += std::sqrt(m[i]) * vsig * (rho * pn + rhoc * vn);
      }
   }
   for (int j = 0; j < 3; ++j)
      p[j] = pfric * p[j] + dp[j];
}

// Reference trajectories from Fortran Tinker 8.10.5 (~/tinker_legacy),
// produced by
//    dynamic.x g3.xyz 10 0.1 0.0001 2 298
// on g3.dyn (no box) or g3pbc.dyn (with the periodic cell's keys), with the
// same RANDOMSEED. Energies come from the log, which prints four decimals;
// the final coordinates come from the last frame of the archive, which prints
// six. Fortran 8.10.5 never removes inertia in stochastic dynamics, so these
// are references for the cells without removal.
static const double g3_pot[2][10] = {
   {17.6071, 17.4366, 17.2742, 17.1252, 16.9497, 16.7513, 16.5434, 16.3543, 16.1759, 16.0185},
   {28.2011, 28.0317, 27.8706, 27.7226, 27.5481, 27.3507, 27.1438, 26.9557, 26.7781, 26.6214}};
static const double g3_kin[2][10] = {
   {15.0109, 15.3748, 16.2659, 16.2372, 16.6306, 17.4837, 15.8362, 15.7196, 16.0082, 15.2176},
   {15.0099, 15.3722, 16.2621, 16.2334, 16.6251, 17.4758, 15.8277, 15.7110, 15.9995, 15.2080}};
static const double g3_xyz10[2][20][3] = {
   {{2.251125, -8.674469, -9.320083}, {3.056947, -7.238430, -13.687479},
      {4.472119, -7.221294, -13.394276}, {2.353982, -7.439405, -12.359208},
      {4.695021, -8.336345, -12.398479}, {3.368089, -8.412901, -11.617271},
      {3.416223, -8.135841, -10.116180}, {2.449207, -8.793252, -8.091180},
      {1.297022, -8.949186, -10.075506}, {2.776613, -6.290624, -14.100571},
      {2.726457, -7.983473, -14.406847}, {5.060731, -7.467839, -14.299131},
      {4.810927, -6.263950, -13.015115}, {2.247034, -6.507958, -11.805922},
      {1.388113, -7.928338, -12.532347}, {4.875142, -9.382288, -12.836548},
      {5.596862, -8.114126, -11.730311}, {2.999723, -9.408701, -11.739652},
      {3.443032, -7.015822, -10.056027}, {4.379849, -8.462221, -9.669014}},
   {{2.251129, -8.674468, -9.320085}, {3.056945, -7.238431, -13.687476},
      {4.472120, -7.221295, -13.394274}, {2.353981, -7.439404, -12.359208},
      {4.695022, -8.336346, -12.398480}, {3.368088, -8.412899, -11.617274},
      {3.416224, -8.135841, -10.116179}, {2.449201, -8.793259, -8.091161},
      {1.297020, -8.949187, -10.075505}, {2.776631, -6.290613, -14.100654},
      {2.726485, -7.983442, -14.406907}, {5.060735, -7.467815, -14.299181},
      {4.810960, -6.263924, -13.015227}, {2.247038, -6.507959, -11.805928},
      {1.388118, -7.928335, -12.532348}, {4.875140, -9.382286, -12.836548},
      {5.596858, -8.114121, -11.730314}, {2.999729, -9.408699, -11.739657},
      {3.443032, -7.015821, -10.056028}, {4.379852, -8.462220, -9.669016}}};

// Tolerances, relative to the scales in Momenta, set at about ten times the
// largest deviation seen over five runs each of the CUDA mixed, OpenACC double
// and double builds. They are checked as |a - b| < eps * scale rather than
// with COMPARE_REALS, whose Approx keeps a hidden relative epsilon of about
// 1.2e-5 that would be looser than most of them.
//
// eps_conserved_p, eps_predicted: rounding in the forces, which sum to zero
//    only to about 3e-8 in mixed precision and 4e-15 in double.
// eps_conserved_l: velocity Verlet would conserve the angular momentum
//    exactly, but mutual polarization converged to POLAR-EPS 1e-5 leaves a
//    small net torque, 1.5e-7 even in double precision.
// eps_removed: what is left after removal, at round-off (2e-15).
// eps_present: what there is to remove, at least 0.037 in every cell.
// eps_recorded: the spread of the recorded momenta, 3e-8 in mixed precision
//    and 6e-15 in double.
const double eps_conserved_p = testGetEps(3e-7, 5e-14);
const double eps_conserved_l = 2e-6;
const double eps_predicted = testGetEps(2e-7, 5e-14);
const double eps_removed = 1e-12;
const double eps_present = 1e-3;
const double eps_recorded = testGetEps(3e-7, 6e-14);

// Linear and angular momentum at steps 1 to 10 of the Momentum section in the
// cells without removal, recorded from the double build. The angular momentum
// is only recorded without a box, where it is defined.
static const double g3_pref[2][10][3] = {
   {{6.292356047887919e+00, 1.942459716439433e+02, -6.449988559682436e+01},
      {-4.095577129970604e+00, 1.711242505731304e+02, -5.296123813907399e+01},
      {5.597236279446062e+01, 1.531940435625885e+02, -4.560716198465757e+01},
      {6.348230965511911e+01, 1.929830513502438e+02, -3.760052788577121e+01},
      {-1.342115959335892e+00, 1.877345691545577e+02, -5.262422348213064e+01},
      {9.385286194930346e+00, 2.295021333154885e+02, -3.051615225627631e+01},
      {-6.015243057413127e+00, 1.970847548126637e+02, 2.413036340275481e+00},
      {-8.335099314951486e+01, 1.760852977659171e+02, 2.795937270305904e+01},
      {-8.858682793539242e+01, 1.751066427513719e+02, 2.112591713522395e+01},
      {-4.649288300076118e+01, 1.662097185061717e+02, -3.212883045217426e+01}},
   {{6.292356047888511e+00, 1.942459716439432e+02, -6.449988559682382e+01},
      {-4.095577129969660e+00, 1.711242505731302e+02, -5.296123813907355e+01},
      {5.597236279446102e+01, 1.531940435625888e+02, -4.560716198465742e+01},
      {6.348230965511912e+01, 1.929830513502450e+02, -3.760052788577025e+01},
      {-1.342115959335651e+00, 1.877345691545584e+02, -5.262422348212935e+01},
      {9.385286194930169e+00, 2.295021333154879e+02, -3.051615225627548e+01},
      {-6.015243057414931e+00, 1.970847548126619e+02, 2.413036340275776e+00},
      {-8.335099314951751e+01, 1.760852977659150e+02, 2.795937270305980e+01},
      {-8.858682793539451e+01, 1.751066427513699e+02, 2.112591713522581e+01},
      {-4.649288300076282e+01, 1.662097185061696e+02, -3.212883045217189e+01}}};
static const double g3_lref[10][3] = {
   {-2.611670950434548e+02, -1.079328690961388e+02, -4.855200268916826e+02},
   {-1.658415461389196e+02, -8.620367101914283e+00, -3.758112716280065e+02},
   {-3.270429622037592e+02, -1.534286521114674e+02, -4.956753371671011e+02},
   {-2.399550948190854e+02, -1.617379271520110e+02, -4.878515145823746e+02},
   {-1.980749497316536e+02, -2.111022513572725e+02, -4.562270396496586e+02},
   {-3.281575424274277e+02, -2.236610063869967e+02, -5.208432184713657e+02},
   {-3.007467673676916e+02, -2.975983693886831e+02, -4.833585211883938e+02},
   {-2.224567344865810e+02, -3.320172522520818e+02, -4.463897445046440e+02},
   {-1.649266366916761e+02, -3.543241055484908e+02, -4.533259908098660e+02},
   {-2.632370012357575e+02, -2.953431495370333e+02, -4.647381574910434e+02}};

void sdCellTests(Cell c)
{
   const int natom = 20;
   const int nfree_full = 3 * natom;
   const int nfree_rest = 3 * natom - (c.pbc ? 3 : 6);

   SECTION("Setup")
   {
      // REMOVE-INERTIA N turns removal on every N steps, and the removed
      // modes come out of the degrees of freedom; without it nothing is
      // removed and all 3N modes are counted. An explicit DEGREES-FREEDOM is
      // taken as given either way.
      for (bool df : {false, true}) {
         CAPTURE(df);
         const char* k = "test_g3.key";
         const char* x = "test_g3.xyz";
         TestFile fke(TINKER9_DIRSTR "/test/file/stochastic/g3.key", k, //
            cellKey(c) + (df ? "\nDEGREES-FREEDOM 50\n" : ""));
         TestFile fx(TINKER9_DIRSTR "/test/file/stochastic/g3.xyz", x);
         TestFile fp(TINKER9_DIRSTR "/test/file/stochastic/hostsG3.prm");
         const char* argv[] = {"dummy", x};
         int argc = 2;
         TestSession session(argc, argv);
         testMdInit(298., 0.);
         COMPARE_INTS(mdstuf::dorest, (c.rest ? 1 : 0));
         COMPARE_INTS(mdstuf::irest, (c.rest ? cell_irest : 0));
         COMPARE_INTS(mdstuf::nfree, (df ? 50 : (c.rest ? nfree_rest : nfree_full)));
         session.end();
         bath::kelvin = 0.;
         bath::isothermal = 0;
      }
   }

   SECTION("GuardRails")
   {
      // The constructor is what validates the SD keywords. In a real run it is
      // reached through initialize(), which builds the integrator named by the
      // key file, so an unsupported option aborts setup rather than being
      // silently ignored. Here the integrator is built directly on top of a
      // plain Verlet setup: throwing out of initialize() would leave finish()
      // to tear down resources that were never allocated, which is a
      // pre-existing hazard in the harness and not what this test is about.
      const char* k = "test_g3.key";
      const char* d = "test_g3.dyn";
      const char* x = "test_g3.xyz";
      TestFile fke(TINKER9_DIRSTR "/test/file/stochastic/g3.key", k, //
         cellKey(c) + "\nINTEGRATOR VERLET\nFRICTION 1.0\n");
      TestFile fd(c.pbc ? TINKER9_DIRSTR "/test/file/stochastic/g3pbc.dyn" //
                        : TINKER9_DIRSTR "/test/file/stochastic/g3.dyn",
         d);
      TestFile fx(TINKER9_DIRSTR "/test/file/stochastic/g3.xyz", x);
      TestFile fp(TINKER9_DIRSTR "/test/file/stochastic/hostsG3.prm");

      const char* argv[] = {"dummy", x};
      int argc = 2;
      TestSession session(argc, argv);
      testMdInit(298., 0.);
      rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md;
      session.init();

      auto constructThrows = [] {
         bool threw = false;
         try {
            StochasticIntegrator intg;
         } catch (...) {
            threw = true;
         }
         return threw;
      };

      // the ordinary NVT case has to work
      REQUIRE(stodyn::friction == Approx(1.0));
      REQUIRE(constructThrows() == false);

      // FRICTION-SCALING scales gamma by each atom's solvent-accessible
      // surface area. It is not ported, so it must be rejected, not quietly
      // ignored.
      stodyn::use_sdarea = 1;
      REQUIRE(constructThrows() == true);
      stodyn::use_sdarea = 0;

      // A negative friction would give an imaginary noise amplitude.
      stodyn::friction = -1.0;
      REQUIRE(constructThrows() == true);
      stodyn::friction = 1.0;

      // Fortran sdstep rescales only the configurational virial and ignores
      // the work done by the friction and random forces, which is why Fortran
      // mdstat suppresses the pressure column for SD. NPT is refused rather
      // than reporting a pressure that is not right.
      bath::isobaric = 1;
      REQUIRE(constructThrows() == true);
      bath::isobaric = 0;

      session.end();
      bath::kelvin = 0.;
      bath::isothermal = 0;
   }

   SECTION("ZeroFrictionIsVelocityVerlet")
   {
      // With zero friction the propagator collapses to plain velocity Verlet:
      // pfric = 1, vfric = dt, afric = dt*dt/2, and no random terms at all.
      // The trajectory is then fully deterministic and must reproduce
      // VerletIntegrator step for step, including the removal steps, since
      // REMOVE-INERTIA applies to both. This is the tightest check of the
      // coefficient plumbing, the kernel indexing and the gradient-buffer
      // convention that can be made without any statistics.
      //
      // Velocity Verlet also conserves the linear momentum and, without a box,
      // the angular momentum, so both must stay at their starting values, or
      // at zero once they have been removed.
      const int nsteps = 10;
      const double dt = 0.001;
      std::vector<double> vpot, vkin, spot, skin;
      std::vector<Momenta> smom;

      // without the keyword, mdinit would turn removal on every 100 steps for
      // Verlet; say "off" explicitly so that it matches the SD run
      std::string vkey = c.rest ? "\nINTEGRATOR VERLET\n" : "\nINTEGRATOR VERLET\nREMOVE-INERTIA 0\n";
      runG3(c, vkey, 0., nsteps, dt,
         [] { return new VerletIntegrator(ThermostatEnum::NONE, BarostatEnum::NONE); },
         [&](int i) {
            if (i > 0) {
               vpot.push_back(esum);
               vkin.push_back(eksum);
            }
         });

      auto s = runG3(c, "\nFRICTION 0.0\n", 0., nsteps, dt,
         [] { return new StochasticIntegrator; },
         [&](int i) {
            if (i > 0) {
               spot.push_back(esum);
               skin.push_back(eksum);
            }
            smom.push_back(measureMomenta());
         });

      // The two integrators evaluate the same expressions in different
      // kernels; in mixed precision that leaves up to 2e-5 kcal/mol between
      // them, in double 1e-12. epsilon(0) drops the hidden relative tolerance
      // of Approx, which would otherwise allow about 3e-4 here.
      const double eps_zf = testGetEps(2e-4, 1e-11);
      COMPARE_INTS(s.natom, natom);
      COMPARE_INTS((int)vpot.size(), nsteps);
      COMPARE_INTS((int)spot.size(), nsteps);
      for (int i = 0; i < nsteps; ++i) {
         CAPTURE(i + 1);
         REQUIRE(spot[i] == Approx(vpot[i]).epsilon(0).margin(eps_zf));
         REQUIRE(skin[i] == Approx(vkin[i]).epsilon(0).margin(eps_zf));
      }

      bool removed = false;
      for (int i = 1; i <= nsteps; ++i) {
         CAPTURE(i);
         removed = removed or removedAt(c, i);
         const auto& a = smom[i];
         const auto& a0 = smom[0];
         for (int j = 0; j < 3; ++j) {
            CAPTURE(j);
            double pref = removed ? 0 : a0.p[j];
            REQUIRE(std::fabs(a.p[j] - pref) < eps_conserved_p * a.pscale);
            if (not c.pbc) {
               double lref = removed ? 0 : a0.l[j];
               REQUIRE(std::fabs(a.l[j] - lref) < eps_conserved_l * a.lscale);
            }
         }
      }
   }

   SECTION("Momentum")
   {
      // A real stochastic run with Philox noise. The linear momentum is
      // checked at every step against advanceMomentum(), which predicts it
      // from the noise alone, and is reset to zero at each removal step.
      // Without a box, the angular momentum must be zero after each removal;
      // with no removal it is checked against a recorded value instead, since
      // it depends on the positions and cannot be predicted from the noise.
      // Under periodic boundaries the angular momentum is neither removed nor
      // well defined, so it is not checked.
      const int nsteps = 10;
      const double dt = 0.001;
      const unsigned int seed = 7;
      std::vector<Momenta> mom;
      std::vector<std::array<double, 3>> pref;
      std::vector<double> m(natom);

      double p[3];
      runG3(c, "\nFRICTION 20.0\nRANDOMSEED 7\n", 298., nsteps, dt,
         [] { return new StochasticIntegrator; },
         [&](int i) {
            auto a = measureMomenta();
            if (i == 0) {
               darray::copyout(g::q0, natom, m.data(), mass);
               waitFor(g::q0);
               for (int j = 0; j < 3; ++j)
                  p[j] = a.p[j];
            } else {
               advanceMomentum(p, i, seed, m);
               if (removedAt(c, i))
                  p[0] = p[1] = p[2] = 0;
            }
            mom.push_back(a);
            pref.push_back({p[0], p[1], p[2]});
         });

      COMPARE_INTS((int)mom.size(), nsteps + 1);
      for (int i = 1; i <= nsteps; ++i) {
         CAPTURE(i);
         const auto& a = mom[i];
         for (int j = 0; j < 3; ++j) {
            CAPTURE(j);
            REQUIRE(std::fabs(a.p[j] - pref[i][j]) < eps_predicted * a.pscale);
         }
         if (removedAt(c, i)) {
            REQUIRE(norm3(a.p) < eps_removed * a.pscale);
            if (not c.pbc)
               REQUIRE(norm3(a.l) < eps_removed * a.lscale);
         } else {
            // removal is real only if there was something to remove
            REQUIRE(norm3(a.p) > eps_present * a.pscale);
            if (not c.pbc)
               REQUIRE(norm3(a.l) > eps_present * a.lscale);
         }
      }

      if (not c.rest) {
         const int ib = c.pbc ? 1 : 0;
         for (int i = 1; i <= nsteps; ++i) {
            CAPTURE(i);
            const auto& a = mom[i];
            for (int j = 0; j < 3; ++j) {
               CAPTURE(j);
               REQUIRE(std::fabs(a.p[j] - g3_pref[ib][i - 1][j]) < eps_recorded * a.pscale);
               if (not c.pbc)
                  REQUIRE(std::fabs(a.l[j] - g3_lref[i - 1][j]) < eps_recorded * a.lscale);
            }
         }
      }
   }

   SECTION("TrajectoryVsTinker")
   {
      // Step-by-step comparison of a stochastic trajectory against Fortran
      // Tinker, restarted from a .dyn file so that neither code draws random
      // numbers to build initial velocities.
      //
      // Two keywords are added to the deck that produced the reference:
      //
      //   FRICTION 91.0    Fortran Tinker 8.10.5 defaults the friction to
      //                    91/ps, while the Fortran bundled here defaults to
      //                    0.5/ps (91.0 only with an implicit solvent). The
      //                    reference was generated with the former, so it has
      //                    to be said out loud rather than relied on.
      //
      //   SD-NOISE TINKER  Draw the random terms from Fortran "normal" in the
      //                    order Fortran "sdterm" draws them, instead of from
      //                    the counter-based generator the integrator normally
      //                    uses. Both codes then consume the same stream from
      //                    the same RANDOMSEED, which is what makes the
      //                    trajectories comparable at all -- any other
      //                    generator would diverge from the first step no
      //                    matter how correct it was.
      //
      // The reference never removes inertia. Removal does not change the
      // energy of the step it happens on, so the removal cells match it up to
      // the first removal. Under periodic boundaries removing the translation
      // only shifts every atom by the same amount, which changes neither the
      // forces nor the noise, so there the potential energy matches at every
      // step and the final structure matches up to a common offset.
      const int nsteps = 10;
      const double dt = 0.0001; // ps
      const int ib = c.pbc ? 1 : 0;

      std::vector<double> pot, kin;
      std::vector<pos_prec> fx1(natom), fy1(natom), fz1(natom);
      runG3(c, "\nFRICTION 91.0\nSD-NOISE TINKER\n", 298., nsteps, dt,
         [] { return new StochasticIntegrator; },
         [&](int i) {
            if (i == 0)
               return;
            pot.push_back(esum);
            kin.push_back(eksum);
            if (i == nsteps) {
               darray::copyout(g::q0, natom, fx1.data(), xpos);
               darray::copyout(g::q0, natom, fy1.data(), ypos);
               darray::copyout(g::q0, natom, fz1.data(), zpos);
               waitFor(g::q0);
            }
         });

      // The reference energies are only printed to four decimals, so half a
      // unit in the last place is already +/- 5e-5 before any real
      // difference. What is left over is the force difference between the two
      // codes, which for this system stays at the 1e-4 level over ten steps.
      const double eps_e = 5e-4;
      const int nsame = c.rest ? cell_irest : nsteps;
      COMPARE_INTS((int)pot.size(), nsteps);
      for (int i = 0; i < nsteps; ++i) {
         CAPTURE(i + 1);
         if (i < nsame or c.pbc)
            COMPARE_REALS(pot[i], g3_pot[ib][i], eps_e);
         if (i < nsame)
            COMPARE_REALS(kin[i], g3_kin[ib][i], eps_e);
      }

      // Energies are averages over the whole system and can hide a single bad
      // atom or a transposed component, so the final structure is checked
      // atom by atom as well, up to the common offset that removing the
      // translation leaves under periodic boundaries.
      if (c.rest and not c.pbc)
         return;
      double off[3] = {0};
      if (c.rest) {
         for (int i = 0; i < natom; ++i) {
            off[0] += (fx1[i] - g3_xyz10[ib][i][0]) / natom;
            off[1] += (fy1[i] - g3_xyz10[ib][i][1]) / natom;
            off[2] += (fz1[i] - g3_xyz10[ib][i][2]) / natom;
         }
      }
      // The archive prints six decimals, and the two codes agree to that
      // rounding (5e-7). epsilon(0) drops the hidden relative tolerance of
      // Approx, which would otherwise allow about 2e-4 Angstrom here.
      const double eps_x = 1e-5;
      for (int i = 0; i < natom; ++i) {
         CAPTURE(i);
         REQUIRE((double)fx1[i] - off[0] == Approx(g3_xyz10[ib][i][0]).epsilon(0).margin(eps_x));
         REQUIRE((double)fy1[i] - off[1] == Approx(g3_xyz10[ib][i][1]).epsilon(0).margin(eps_x));
         REQUIRE((double)fz1[i] - off[2] == Approx(g3_xyz10[ib][i][2]).epsilon(0).margin(eps_x));
      }
   }
}
}

TEST_CASE("SD-NoPbc-NoRemoveInertia", "[md][stochastic][g3]")
{
   sdCellTests({false, false});
}

TEST_CASE("SD-NoPbc-RemoveInertia", "[md][stochastic][g3]")
{
   sdCellTests({false, true});
}

TEST_CASE("SD-Pbc-NoRemoveInertia", "[md][stochastic][g3]")
{
   sdCellTests({true, false});
}

TEST_CASE("SD-Pbc-RemoveInertia", "[md][stochastic][g3]")
{
   sdCellTests({true, true});
}

TEST_CASE("SD-TinkerRandom", "[md][stochastic]")
{
   // The SD-NOISE TINKER path reproduces Tinker's own generator rather than
   // calling the Fortran one, so that the stream does not depend on what else
   // in the process drew from it first. These values come from an independent
   // transcription of random.f and normal() in random.f; the end-to-end check
   // that the stream really matches Fortran is the TrajectoryVsTinker section
   // of the SD-Pbc-* and SD-NoPbc-* cases.
   const double eps = 1e-14;

   SECTION("UniformStream")
   {
      static const double ref[] = {0.285380899094686, 0.253358189265917, 0.093468531009194,
         0.608496890739648, 0.903420260078610, 0.195873192813816};
      double got[6];
      sdTinkerRandomSample(1, false, 6, got);
      for (int i = 0; i < 6; ++i) {
         CAPTURE(i);
         REQUIRE(got[i] == Approx(ref[i]).margin(eps));
      }
   }

   SECTION("NormalStream")
   {
      // Marsaglia polar, which caches the second deviate of each pair, so the
      // odd and even draws exercise different branches.
      static const double ref[] = {-0.983377463093775, -0.855700768460867, 0.214221269465584,
         -0.802674498515331, 0.708852264118617, -0.059815920369672};
      double got[6];
      sdTinkerRandomSample(1, true, 6, got);
      for (int i = 0; i < 6; ++i) {
         CAPTURE(i);
         REQUIRE(got[i] == Approx(ref[i]).margin(eps));
      }
   }

   SECTION("SeedIsHonoured")
   {
      static const double ref[] = {
         0.347349908618421, -0.664738299694116, -1.337153479738589, 1.925530978941072};
      double got[4];
      sdTinkerRandomSample(12345, true, 4, got);
      for (int i = 0; i < 4; ++i) {
         CAPTURE(i);
         REQUIRE(got[i] == Approx(ref[i]).margin(eps));
      }
   }
}
