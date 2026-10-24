#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/induce.h"
#include "ff/amoeba/mpole.h"
#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/empole.h"
#include "ff/hippo/epolar.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/potent.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <tinker/detail/atoms.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace tinker;

#if TINKER_GPULANG_CUDA

// The energy terms leave pole and rpole in whatever state they last needed.
// mpoleEnsurePhysical() must bring rpole to the physical multipoles, which the
// saved dipoles and the analyze moments read: pole at the electrostatic lambda,
// with charge flux, rotated at the current coordinates. The reference is Tinker
// rotating its own pole, which altelec left at the electrostatic lambda, at the
// same coordinates.

namespace {
struct Fixture
{
   const char* name;
   const char* xyz;
   const char* key;
   const char* prm;
   const char* extra = "";    // key lines the fixture adds to the file
   void (*terms)() = nullptr; // checks that the setup is the one the fixture is about
};

// Free energy perturbation of a ligand with charge flux, whose polarization
// is solved with every parameter scaled to the polarization lambda.
const char* kCfluxFep = "\nligand -1 3\nvdw-lmda-map exp\nvdw-lmda-exp 1\nele-lmda-map exp\nele-lmda-exp 1\n"
                        "pol-lmda-map exp\npol-lmda-exp 1\nlambda 0.5\n"
                        "lambda-mode fep\nlambda-window 0.5\nlambda-window 0.4\n";

void termsCfluxFep()
{
   REQUIRE(use_fep);
   REQUIRE(use_prst);
   REQUIRE(use(Potent::CHGFLX));
}

const std::string kDir = TINKER9_DIRSTR "/test/file/";
const char* kWater03 = "commit_6fe8e913/water03.prm";

const Fixture kFixtures[] = {
   // single topology polarization at plambda != elambda, no lambda derivative
   {"abs-prng-nodl", "mutate/water2.xyz", "mpolestate/abs_prng_nodl.key", kWater03},
   // the same with the lambda derivative, whose last pass masks rpole
   {"abs-prng-dl", "mutate/water2.xyz", "mpolestate/abs_prng_dl.key", kWater03},
   // the fused multipole and polarization single topology
   {"emplarast", "mutate/water2.xyz", "mutate/142_water_rels_st_l085.key", kWater03},
   // dual topology polarization, whose passes rotate poleorig
   {"epdt", "mutate/water2.xyz", "mutate/083_water_rels_ye_l085.key", kWater03},
   // multipoles scaled on the fly from poleorig, no polarization
   {"emast-mponly", "mutate/water2.xyz", "mpolestate/emast_mponly.key", kWater03},
   // electrostatics driven by the main lambda, polarization pinned
   {"polpinned", "mutate/water2.xyz", "mpolestate/polpinned.key", kWater03},
   // charge flux rewrites the monopoles of pole
   {"hippo-cflux", "water10/h2o10.xyz", "mpolestate/hippo_cflux.key", "hippo/water19.prm"},
   // the same under free energy perturbation, which scales pole between calls
   {"hippo-cflux-fep", "water10/h2o10.xyz", "mpolestate/hippo_cflux.key", "hippo/water19.prm", kCfluxFep,
      termsCfluxFep},
   // staged at lambda 1, where single topology polarization at plambda equal
   // to elambda lets empole and epolar, run apart, share cmp
   {"rels-l100", "mutate/water2.xyz", "mpolestate/rels_l100.key", kWater03},
   // all three staged legs under one main lambda, single topology
   {"rels3-st", "mutate/water2.xyz", "mutate/196_water_rels3_st_l085.key", kWater03},
};

// The fixture of a given name, so that editing the list shifts no case.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no mpolestate fixture named " << name);
   return kFixtures[0];
}

// max |rpole - Tinker's rpole| at the coordinates on the device
double rpoleError()
{
   std::vector<pos_prec> xb(n), yb(n), zb(n);
   std::vector<real> rp(n * MPL_TOTAL);
   darray::copyout(g::q0, n, xb.data(), xpos);
   darray::copyout(g::q0, n, yb.data(), ypos);
   darray::copyout(g::q0, n, zb.data(), zpos);
   darray::copyout(g::q0, n * MPL_TOTAL, rp.data(), &rpole[0][0]);
   waitFor(g::q0);
   for (int i = 0; i < n; ++i) {
      atoms::x[i] = xb[i];
      atoms::y[i] = yb[i];
      atoms::z[i] = zb[i];
   }
   if (use(Potent::CHGFLX))
      tinker_f_alterchg();
   tinker_f_chkpole();
   tinker_f_rotpole(tinker_fchars{const_cast<char*>("MPOLE"), 5});

   // Tinker c = 0, dx = 1, dy = 2, dz = 3, qxx = 4, qxy = 5, qxz = 6, qyy = 8, qyz = 9, qzz = 12
   const int map[10][2] = {{MPL_PME_0, 0}, {MPL_PME_X, 1}, {MPL_PME_Y, 2}, {MPL_PME_Z, 3}, {MPL_PME_XX, 4},
      {MPL_PME_XY, 5}, {MPL_PME_XZ, 6}, {MPL_PME_YY, 8}, {MPL_PME_YZ, 9}, {MPL_PME_ZZ, 12}};
   double worst = 0;
   for (int i = 0; i < n; ++i)
      for (const auto& m : map)
         worst = std::max(worst, std::fabs(rp[MPL_TOTAL * i + m[0]] - mpole::rpole[mpole::maxpole * i + m[1]]));
   return worst;
}

void runFixture(const Fixture& fx, int vers)
{
   TestLmdaFlagReset lmdaReset;
   std::string xyzname = "mpolestate.xyz";
   std::string keyname = "mpolestate.key";
   std::string prm = fx.prm;
   std::string prmname = prm.substr(prm.find('/') + 1);
   TestFile fxyz(kDir + fx.xyz, xyzname);
   TestFile fkey(kDir + fx.key, keyname, fx.extra);
   TestFile fprm(kDir + prm, prmname);
   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   const double eps = testGetEps(1.0e-5, 1.0e-10);

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   TestSession session(argc, argv);
   session.init();
   if (fx.terms)
      fx.terms();

   energy(vers);
   mpoleEnsurePhysical();
   COMPARE_REALS(rpoleError(), 0, eps);

   // Move every atom without evaluating the energy, as a rejected Monte Carlo
   // volume move does before the coordinates are saved.
   std::vector<pos_prec> xb(n), yb(n), zb(n);
   darray::copyout(g::q0, n, xb.data(), xpos);
   darray::copyout(g::q0, n, yb.data(), ypos);
   darray::copyout(g::q0, n, zb.data(), zpos);
   waitFor(g::q0);
   for (int i = 0; i < n; ++i) {
      xb[i] += 0.02 * ((i * 7) % 5 - 2);
      yb[i] += 0.02 * ((i * 3) % 5 - 2);
      zb[i] += 0.02 * ((i * 11) % 5 - 2);
   }
   darray::copyin(g::q0, n, xpos, xb.data());
   darray::copyin(g::q0, n, ypos, yb.data());
   darray::copyin(g::q0, n, zpos, zb.data());
   waitFor(g::q0);
   copyPosToXyz(true);
   mpoleEnsurePhysical();
   COMPARE_REALS(rpoleError(), 0, eps);

   session.end();
}
}

// With use_emast the lambda derivative grid is opened as a private unit with
// the multipole parameters; polarization must still share the multipole unit,
// which the fused kernels assume, rather than land on the private one.
TEST_CASE("MPOLESTATE-pme-units", "[ff][mpolestate]")
{
   const Fixture& f = fx("abs-prng-dl");
   std::string xyzname = "mpolestate.xyz";
   std::string keyname = "mpolestate.key";
   TestFile fxyz(kDir + f.xyz, xyzname);
   TestFile fkey(kDir + f.key, keyname);
   TestFile fprm(kDir + f.prm, "water03.prm");
   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   TestSession session(argc, argv);
   session.init();

   REQUIRE(use_emast);
   REQUIRE(dlpme_unit.valid());
   REQUIRE(ppme_unit == epme_unit);
   REQUIRE(dlpme_unit != epme_unit);
   REQUIRE(pvpme_unit != ppme_unit);

   session.end();
}

TEST_CASE("MPOLESTATE-physical", "[ff][mpolestate]")
{
   for (const auto& fx : kFixtures) {
      for (int vers : {calc::v0, calc::v1, calc::v4}) {
         CAPTURE(fx.name, vers);
         runFixture(fx, vers);
      }
   }
}

namespace {
// Sets a fixture up, and keeps its files, for as long as it is in scope. The
// session is declared after the files, so it ends before they are removed.
const char* kSetupArgv[] = {"dummy", "mpolestate.xyz", "-k", "mpolestate.key"};

struct Setup
{
   TestLmdaFlagReset lmdaReset;
   TestFile fxyz, fkey, fprm;
   TestSession session;

   Setup(const Fixture& f, int rc)
      : fxyz(kDir + f.xyz, "mpolestate.xyz")
      , fkey(kDir + f.key, "mpolestate.key", f.extra)
      , fprm(kDir + f.prm, std::string(f.prm).substr(std::string(f.prm).find('/') + 1))
      , session(4, kSetupArgv)
   {
      rc_flag = rc;
      session.init();
      if (f.terms)
         f.terms();
   }
};

const int kEnergyGrad = calc::xyz | calc::mass | calc::energy | calc::grad;

std::vector<real> grab(real (*dev)[3])
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real(*)[3]>(v.data()), dev);
   waitFor(g::q0);
   return v;
}

double maxDiff(const std::vector<real>& a, const std::vector<double>& b)
{
   double worst = 0;
   for (size_t i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::fabs(a[i] - b[i]));
   return worst;
}

// Two solves of one state agree to a few 1e-9 e*Angstrom here, far inside the
// POLAR-EPS 1e-5 they are converged to; a wrong state or weighting is off by
// 1e-2 or more.
const double kSolveEps = 1.0e-6;
}

// Single topology: the reported dipoles are the solve at plambda, which is the
// one epolar() leaves behind.
TEST_CASE("MPOLESTATE-induced-prst", "[ff][mpolestate]")
{
   Setup s(fx("abs-prng-dl"), kEnergyGrad);
   REQUIRE(use_prst);
   REQUIRE_FALSE(lmdaSameValue(elam, plam));

   energy(calc::v4);
   const auto ref = grab(uind);
   epolarPhysicalInduced();
   const auto got = grab(uind);
   COMPARE_REALS(maxDiff(got, std::vector<double>(ref.begin(), ref.end())), 0, kSolveEps);
}

// The active leg of a staged relative schedule picks the ligand that charges,
// so a sub-lambda reached again on another leg is another state of pole even
// though the number is the same.
TEST_CASE("MPOLESTATE-leg-jump", "[ff][mpolestate]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-6);

   Setup s(fx("rels3-st"), kEnergyGrad);
   REQUIRE(use_prst);

   struct State
   {
      double weight; // the polarization lambda
      double e;
      std::vector<real> u;
   };
   auto evalAt = [](double lmda, RelStage leg) {
      lambda = lmda;
      energy(calc::v4);
      REQUIRE(relstage == leg);
      return State{plam, esum, grab(uind)};
   };
   auto same = [&](const State& got, const State& ref) {
      COMPARE_REALS(got.e, ref.e, eps);
      COMPARE_REALS(maxDiff(got.u, std::vector<double>(ref.u.begin(), ref.u.end())), 0, kSolveEps);
   };

   // Reference states, each reached through the middle leg, whose zero weight
   // neither charging leg shares.
   const State lig1 = evalAt(0.85, RelStage::LIG1);
   evalAt(0.5, RelStage::VDWM);
   const State lig2 = evalAt(0.15, RelStage::LIG2);
   // The mirror image of the main lambda carries the same weight on the other
   // ligand, or the jumps below would test nothing.
   REQUIRE(lig2.weight == lig1.weight);
   REQUIRE(std::fabs(lig2.e - lig1.e) > 100 * eps);

   // Straight from one charging leg to the other, both ways.
   same(evalAt(0.85, RelStage::LIG1), lig1);
   same(evalAt(0.15, RelStage::LIG2), lig2);
}

// Dual topology leaves only its last endpoint pass in uind and udir, so it has
// no dipoles to report.
TEST_CASE("MPOLESTATE-induced-epdt", "[ff][mpolestate]")
{
   Setup s(fx("epdt"), kEnergyGrad);
   REQUIRE(use_epdt);

   energy(calc::v4);
   REQUIRE_THROWS(epolarPhysicalInduced());
}

namespace {
// What epolar adds to the electrostatic accumulators, and the dipoles it solves.
struct EpResult
{
   double e;
   double v[9];
   std::vector<real> u;
};

enum class Between
{
   KEEP,       // leave what empole built for epolar to reuse
   CLOBBER,    // report cmp overwritten, so that epolar rebuilds all of it
   POISON_PHI, // zero cphi behind the state's back
   POISON_VIR, // zero vir_m behind the state's back
};

// empole then epolar, as energy() runs them apart, with something done between.
// Outside analysis the terms add straight into the electrostatic accumulators,
// so those are cleared between the two to see epolar alone.
EpResult runEmEp(int vers, Between b)
{
   if (mplpot::use_chgpen)
      empoleChgpen(vers);
   else
      empole(vers);
   switch (b) {
   case Between::CLOBBER:
      mpoleCmpClobbered();
      break;
   case Between::POISON_PHI:
      darray::zero(g::q0, n, cphi);
      break;
   case Between::POISON_VIR:
      darray::zero(g::q0, bufferSize(), vir_m);
      break;
   default:
      break;
   }
   darray::zero(g::q0, bufferSize(), eng_buf_elec, vir_buf_elec);
   if (mplpot::use_chgpen)
      epolarChgpen(vers);
   else
      epolar(vers);

   EpResult r;
   r.e = energyReduce(eng_buf_elec);
   virial_prec v[9];
   virialReduce(v, vir_buf_elec);
   for (int i = 0; i < 9; ++i)
      r.v[i] = v[i];
   r.u = grab(uind);
   return r;
}

double relDiff(double a, double b)
{
   return std::fabs(a - b) / std::max(1.0, std::fabs(b));
}

double virDiff(const EpResult& a, const EpResult& b)
{
   double worst = 0;
   for (int i = 0; i < 9; ++i)
      worst = std::max(worst, relDiff(a.v[i], b.v[i]));
   return worst;
}
}

// empole leaves the reciprocal potential of cmp, and with the virial the
// convolution virial in vir_m, for the polarization direct field to reuse. The
// reuse must depend on whether this call computes the virial, not on whether
// vir_m is allocated, and must agree with a rebuild.
TEST_CASE("MPOLESTATE-recip-reuse", "[ff][mpolestate]")
{
   // Grid spreading adds floats atomically, so a rebuild is not bit identical.
   const double eps = testGetEps(1.0e-4, 1.0e-6);

   for (const char* name : {"rels-l100", "hippo-cflux", "hippo-cflux-fep"}) {
      for (int vers : {calc::v4, calc::v1, calc::v6}) {
         CAPTURE(name, vers);
         Setup s(fx(name), kEnergyGrad | calc::virial);
         REQUIRE(vir_m != nullptr);
         REQUIRE(ppme_unit == epme_unit);
         REQUIRE(elam == plam);
         const bool do_e = vers & calc::energy;
         const bool do_v = vers & calc::virial;

         // What empole leaves behind answers a reader that needs the virial
         // only if empole computed one.
         if (mplpot::use_chgpen)
            empoleChgpen(vers);
         else
            empole(vers);
         REQUIRE(mpoleFphiCurrent(ppme_unit, false));
         REQUIRE(mpoleFphiCurrent(ppme_unit, true) == do_v);

         const EpResult ref = runEmEp(vers, Between::CLOBBER);
         const EpResult got = runEmEp(vers, Between::KEEP);
         REQUIRE(mpoleRecipVirial() == do_v);
         COMPARE_REALS(maxDiff(got.u, std::vector<double>(ref.u.begin(), ref.u.end())), 0, kSolveEps);
         if (do_e)
            COMPARE_REALS(relDiff(got.e, ref.e), 0, eps);
         if (do_v)
            COMPARE_REALS(virDiff(got, ref), 0, eps);

         // Poisoning what empole left shows that epolar did read it.
         const EpResult phi = runEmEp(vers, Between::POISON_PHI);
         REQUIRE(maxDiff(phi.u, std::vector<double>(ref.u.begin(), ref.u.end())) > 1.0e-3);
         if (do_v) {
            const EpResult vir = runEmEp(vers, Between::POISON_VIR);
            REQUIRE(virDiff(vir, ref) > 1.0e-3);
         }
      }
   }
}

#endif
