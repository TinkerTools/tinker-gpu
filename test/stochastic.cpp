#include "ff/atom.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/stochastic.h"
#include "seq/philox.h"
#include "tool/accasync.h"
#include "tool/darray.h"
#include <cmath>
#include <cstdio>
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
      // SD-NoiseCorrelation below.
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
// One MD run on the 216-atom argon box, driven by whichever integrator the
// caller builds. Mirrors the setup used by test/nh.cpp.
template <class MakeIntegrator, class PerStep>
void runArbox(const char* keyExtra, double kelvin, int nsteps, double dt, //
   MakeIntegrator makeIntegrator, PerStep perStep)
{
   const char* k = "test_arbox.key";
   const char* d = "test_arbox.dyn";
   const char* x = "test_arbox.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k, keyExtra);
   TestFile fd(TINKER9_DIRSTR "/test/file/arbox/arbox.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/arbox/arbox.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(kelvin, 0.);

   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md;
   initialize();

   int old = inform::iwrite;
   inform::iwrite = 1;
   {
      auto* intg = makeIntegrator();
      for (int i = 1; i <= nsteps; ++i) {
         intg->dynamic(i, dt);
         perStep(i);
      }
      delete intg;
   }
   inform::iwrite = old;

   finish();
   testEnd();

   TestRemoveFileOnExit arc("test_arbox.arc");
   bath::kelvin = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}
}

TEST_CASE("SD-ZeroFriction-IsVelocityVerlet", "[md][stochastic][arbox]")
{
   // With zero friction the propagator collapses to plain velocity Verlet:
   // pfric = 1, vfric = dt, afric = dt*dt/2, and no random terms at all. The
   // trajectory is then fully deterministic and must reproduce VerletIntegrator
   // step for step. This is the tightest check of the coefficient plumbing, the
   // kernel indexing and the gradient-buffer convention that can be made
   // without any statistics.
   const int nsteps = 20;
   const double dt = 0.001;
   std::vector<double> vpot, vkin, spot, skin;

   runArbox("\nintegrator verlet\n", 0., nsteps, dt,
      [] { return new VerletIntegrator(ThermostatEnum::NONE, BarostatEnum::NONE); },
      [&](int) { vpot.push_back(esum); vkin.push_back(eksum); });

   runArbox("\nintegrator stochastic\nfriction 0.0\n", 0., nsteps, dt,
      [] { return new StochasticIntegrator; },
      [&](int) { spot.push_back(esum); skin.push_back(eksum); });

   REQUIRE(vpot.size() == (size_t)nsteps);
   REQUIRE(spot.size() == (size_t)nsteps);
   for (int i = 0; i < nsteps; ++i) {
      CAPTURE(i);
      REQUIRE(spot[i] == Approx(vpot[i]).margin(1e-10));
      REQUIRE(skin[i] == Approx(vkin[i]).margin(1e-10));
   }
}

TEST_CASE("SD-GuardRails", "[md][stochastic]")
{
   // The constructor is what validates the SD keywords. In a real run it is
   // reached through initialize(), which builds the integrator named by the key
   // file, so an unsupported option aborts setup rather than being silently
   // ignored. Here the integrator is built directly on top of a plain Verlet
   // setup: throwing out of initialize() would leave finish() to tear down
   // resources that were never allocated, which is a pre-existing hazard in the
   // harness and not what this test is about.
   const char* k = "test_arbox.key";
   const char* d = "test_arbox.dyn";
   const char* x = "test_arbox.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k, //
      "\nintegrator verlet\nfriction 1.0\n");
   TestFile fd(TINKER9_DIRSTR "/test/file/arbox/arbox.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/arbox/arbox.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 0.);
   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md;
   initialize();

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

   // FRICTION-SCALING scales gamma by each atom's solvent-accessible surface
   // area. It is not ported, so it must be rejected, not quietly ignored.
   stodyn::use_sdarea = 1;
   REQUIRE(constructThrows() == true);
   stodyn::use_sdarea = 0;

   // A negative friction would give an imaginary noise amplitude.
   stodyn::friction = -1.0;
   REQUIRE(constructThrows() == true);
   stodyn::friction = 1.0;

   // Fortran sdstep rescales only the configurational virial and ignores the
   // work done by the friction and random forces, which is why Fortran mdstat
   // suppresses the pressure column for SD. NPT is refused rather than
   // reporting a pressure that is not right.
   bath::isobaric = 1;
   REQUIRE(constructThrows() == true);
   bath::isobaric = 0;

   finish();
   testEnd();
   bath::kelvin = 0.;
   bath::isothermal = 0;
}

namespace {
// Reference trajectory from Fortran Tinker 8.10.5, produced by
//    dynamic.x g3.xyz 10 0.1 0.0001 2 298
// on the same restart file with the same RANDOMSEED. Energies come from the
// log, which prints four decimals; the final coordinates come from the last
// frame of the archive, which prints six.
static const double g3_pot[] = {
   17.6071, 17.4366, 17.2742, 17.1252, 16.9497, //
   16.7513, 16.5434, 16.3543, 16.1759, 16.0185};
static const double g3_kin[] = {
   15.0109, 15.3748, 16.2659, 16.2372, 16.6306, //
   17.4837, 15.8362, 15.7196, 16.0082, 15.2176};
static const double g3_xyz10[][3] = {
   {2.251125, -8.674469, -9.320083},    {3.056947, -7.238430, -13.687479},
   {4.472119, -7.221294, -13.394276},   {2.353982, -7.439405, -12.359208},
   {4.695021, -8.336345, -12.398479},   {3.368089, -8.412901, -11.617271},
   {3.416223, -8.135841, -10.116180},   {2.449207, -8.793252, -8.091180},
   {1.297022, -8.949186, -10.075506},   {2.776613, -6.290624, -14.100571},
   {2.726457, -7.983473, -14.406847},   {5.060731, -7.467839, -14.299131},
   {4.810927, -6.263950, -13.015115},   {2.247034, -6.507958, -11.805922},
   {1.388113, -7.928338, -12.532347},   {4.875142, -9.382288, -12.836548},
   {5.596862, -8.114126, -11.730311},   {2.999723, -9.408701, -11.739652},
   {3.443032, -7.015822, -10.056027},   {4.379849, -8.462221, -9.669014}};
}

TEST_CASE("SD-TrajectoryVsTinker-G3", "[md][stochastic][g3]")
{
   // Step-by-step comparison of a real stochastic trajectory against Fortran
   // Tinker: a 20-atom TEMOA host-guest system with AMOEBA mutual polarization,
   // restarted from a .dyn file so that neither code draws random numbers to
   // build initial velocities.
   //
   // Two keywords are added to the deck that produced the reference:
   //
   //   FRICTION 91.0    Fortran Tinker 8.10.5 defaults the friction to 91/ps,
   //                    while the Fortran bundled here defaults to 0.5/ps
   //                    (91.0 only with an implicit solvent). The reference was
   //                    generated with the former, so it has to be said out
   //                    loud rather than relied on.
   //
   //   SD-NOISE TINKER  Draw the random terms from Fortran "normal" in the
   //                    order Fortran "sdterm" draws them, instead of from the
   //                    counter-based generator the integrator normally uses.
   //                    Both codes then consume the same stream from the same
   //                    RANDOMSEED, which is what makes the trajectories
   //                    comparable at all -- any other generator would diverge
   //                    from the first step no matter how correct it was.
   const int nsteps = 10;
   const double dt = 0.0001; // ps

   const char* k = "test_g3.key";
   const char* d = "test_g3.dyn";
   const char* x = "test_g3.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/stochastic/g3.key", k, //
      "\nFRICTION 91.0\nSD-NOISE TINKER\n");
   TestFile fd(TINKER9_DIRSTR "/test/file/stochastic/g3.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/stochastic/g3.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/stochastic/hostsG3.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 0.);

   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md;
   initialize();

   // n is reset by finish(), so keep the atom count for the checks below --
   // otherwise the per-atom loop would quietly iterate zero times
   const int natom = n;
   std::vector<double> pot, kin;
   std::vector<pos_prec> fx1(natom), fy1(natom), fz1(natom);
   int old = inform::iwrite;
   inform::iwrite = 1;
   {
      StochasticIntegrator intg;
      for (int i = 1; i <= nsteps; ++i) {
         intg.dynamic(i, dt);
         pot.push_back(esum);
         kin.push_back(eksum);
      }
      darray::copyout(g::q1, natom, fx1.data(), xpos);
      darray::copyout(g::q1, natom, fy1.data(), ypos);
      darray::copyout(g::q1, natom, fz1.data(), zpos);
      waitFor(g::q1);
   }
   inform::iwrite = old;

   finish();
   testEnd();
   bath::kelvin = 0.;
   bath::isothermal = 0;

   REQUIRE(natom == 20);

   // The reference energies are only printed to four decimals, so half a unit
   // in the last place is already +/- 5e-5 before any real difference. What is
   // left over is the force difference between the two codes, which for this
   // system stays at the 1e-4 level over ten steps.
   const double eps_e = 5e-4;
   for (int i = 0; i < nsteps; ++i) {
      CAPTURE(i);
      REQUIRE(pot[i] == Approx(g3_pot[i]).margin(eps_e));
      REQUIRE(kin[i] == Approx(g3_kin[i]).margin(eps_e));
   }

   // Energies are averages over the whole system and can hide a single bad
   // atom or a transposed component, so the final structure is checked atom by
   // atom as well.
   const double eps_x = 1e-5;
   for (int i = 0; i < natom; ++i) {
      CAPTURE(i);
      REQUIRE((double)fx1[i] == Approx(g3_xyz10[i][0]).margin(eps_x));
      REQUIRE((double)fy1[i] == Approx(g3_xyz10[i][1]).margin(eps_x));
      REQUIRE((double)fz1[i] == Approx(g3_xyz10[i][2]).margin(eps_x));
   }
}

TEST_CASE("SD-TinkerRandom", "[md][stochastic]")
{
   // The SD-NOISE TINKER path reproduces Tinker's own generator rather than
   // calling the Fortran one, so that the stream does not depend on what else
   // in the process drew from it first. These values come from an independent
   // transcription of random.f and normal() in random.f; the end-to-end check
   // that the stream really matches Fortran is SD-TrajectoryVsTinker-G3.
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
