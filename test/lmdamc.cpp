#include "ff/amoeba/induce.h"
#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/eabf.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/eost.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/darray.h"
#include "tool/iofortstr.h"

#include "test.h"
#include "testrt.h"

#include <tinker/detail/bath.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/inform.hh>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Tests of the Monte Carlo propagation of the main lambda for the adaptive
// bias methods (src/dlmda.cpp:lmdaMonte), following the Fortran suite in
// tinker/test/test_lmdamc.f: the keywords, the phase and frequency of the
// trials, the size and reflection of a move, the energy and the induced dipoles
// kept by an accepted move and restored by a rejected one, the step energy that
// no longer describes the system, and short dynamics runs.
//
// The outcome of a move is forced by construction and never by the random
// seed: a false step energy far below or above the true one, a steep mean
// force at a very low temperature, or a tall gaussian under the lambda.
//
// Lambda dynamics needs the CUDA platform (dlmdaData2).

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
// With the analysis buffers the separate multipole and polarization kernels
// run; without them, as in dynamics, the fused ones do. A trial energy has to
// be right on both routes.
const int kRc = calc::xyz | calc::mass | calc::vmask;
const int kRcDyn = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::virial;
const int kMd = kRcDyn | calc::md;

const char* kIonwat = "mutate/lmdamc_ionwat.key";
const char* kNopol = "mutate/lmdamc_nopol.key";

struct BathReset
{
   ~BathReset()
   {
      bath::kelvin = 0.;
      bath::atmsph = 0.;
      bath::isothermal = 0;
      bath::isobaric = 0;
   }
};

// A system set up from a key file under test/file, with the adaptive bias
// method \c mode, as the dynamic program would set it up; the ion and water
// fixture unless another structure is named.
struct Setup
{
   BathReset bathReset;
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   std::string xyz, key;
   const char* argv[4];
   TestFile fxyz, fkey, fprm;
   TestSession session;

   Setup(const char* keyfile, const std::string& mode, int rc, double atm = 0., const std::string& extra = "",
      const char* xyzfile = "mutate/ionwat.xyz", const char* prmfile = "commit_ebe3611e/amoeba09.prm")
      : xyz(testBaseName(xyzfile))
      , key(testBaseName(keyfile))
      , argv{"dummy", xyz.c_str(), "-k", key.c_str()}
      , fxyz(std::string(TINKER9_DIRSTR "/test/file/") + xyzfile, xyz)
      , fkey(std::string(TINKER9_DIRSTR "/test/file/") + keyfile, key, "\nlambda-mode " + mode + "\n" + extra)
      , fprm(std::string(TINKER9_DIRSTR "/test/file/") + prmfile)
      , session(4, argv)
   {
      testMdInit(298., atm);
      rc_flag = rc;
      session.init();
   }
};

template <class T>
std::vector<T> grab1(const T* dev, int len = n)
{
   std::vector<T> v(len);
   darray::copyout(g::q0, len, v.data(), dev);
   waitFor(g::q0);
   return v;
}

std::vector<real> grab3(const real (*dev)[3])
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real (*)[3]>(v.data()), dev);
   waitFor(g::q0);
   return v;
}

std::vector<real> grabRpole()
{
   std::vector<real> v(MPL_TOTAL * n);
   darray::copyout(g::q0, n, reinterpret_cast<real (*)[MPL_TOTAL]>(v.data()), rpole);
   waitFor(g::q0);
   return v;
}

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b)
{
   return a.size() == b.size() and 0 == std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}

template <class T>
double maxDiff(const std::vector<T>& a, const std::vector<T>& b)
{
   REQUIRE(a.size() == b.size());
   double d = 0;
   for (size_t i = 0; i < a.size(); ++i)
      d = std::max(d, std::fabs((double)a[i] - (double)b[i]));
   return d;
}

// The forces in kcal/mol/Ang; the device keeps them in fixed point.
std::vector<double> gradient()
{
   std::vector<double> gxyz(3 * n);
   copyGradient(calc::grad, gxyz.data(), gxyz.data() + n, gxyz.data() + 2 * n);
   return gxyz;
}

// The bias that the energy call of the step put inside esum.
double biasInEsum()
{
   if (use_meta)
      return bgbias;
   if (use_abf)
      return -bostlmda;
   return bgbias - bostlmda;
}

// Everything but the energy that a rejected lambda move has to put back.
struct State
{
   double lambda, elam, plam, vlam;
   energy_prec dl[8];
   double bias[5];
   std::vector<virial_prec> vir;
   std::vector<grad_prec> gx, gy, gz;
   std::vector<real> uind, uinp, udir, udirp;
   int nualt;
   std::vector<std::vector<real>> slots, pslots;

   State()
      : lambda(tinker::lambda)
      , elam(tinker::elam)
      , plam(tinker::plam)
      , vlam(tinker::vlam)
      , dl{dedl, demdl, depdl, devdl, d2edl2, d2emdl2, d2epdl2, d2evdl2}
      , bias{bgbias, bdgdl, bdgdfl, bostlmda, lmdadfdl}
      , vir(tinker::vir, tinker::vir + 9)
      , gx(grab1(tinker::gx))
      , gy(grab1(tinker::gy))
      , gz(grab1(tinker::gz))
      , nualt(tinker::nualt)
   {
      if (use(Potent::POLAR)) {
         uind = grab3(tinker::uind);
         uinp = grab3(tinker::uinp);
         udir = grab3(tinker::udir);
         udirp = grab3(tinker::udirp);
         real(*ds[])[3] = {udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, udalt_08,
            udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15};
         real(*ps[])[3] = {upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07, upalt_08,
            upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15};
         for (int i = 0; i < maxualt; ++i) {
            slots.push_back(grab3(ds[i]));
            pslots.push_back(grab3(ps[i]));
         }
      }
   }
};

void requireSame(const State& a, const State& b)
{
   REQUIRE(a.lambda == b.lambda);
   REQUIRE(a.elam == b.elam);
   REQUIRE(a.plam == b.plam);
   REQUIRE(a.vlam == b.vlam);
   for (int i = 0; i < 8; ++i)
      REQUIRE(a.dl[i] == b.dl[i]);
   for (int i = 0; i < 5; ++i)
      REQUIRE(a.bias[i] == b.bias[i]);
   REQUIRE(sameBits(a.vir, b.vir));
   REQUIRE(sameBits(a.gx, b.gx));
   REQUIRE(sameBits(a.gy, b.gy));
   REQUIRE(sameBits(a.gz, b.gz));
   REQUIRE(sameBits(a.uind, b.uind));
   REQUIRE(sameBits(a.uinp, b.uinp));
   REQUIRE(sameBits(a.udir, b.udir));
   REQUIRE(sameBits(a.udirp, b.udirp));
   REQUIRE(a.nualt == b.nualt);
   REQUIRE(a.slots.size() == b.slots.size());
   for (size_t i = 0; i < a.slots.size(); ++i) {
      REQUIRE(sameBits(a.slots[i], b.slots[i]));
      REQUIRE(sameBits(a.pslots[i], b.pslots[i]));
   }
}

// Adds a bias, so that the bias change of a move is not zero, and returns the
// energy of the current structure including that bias, found as the force call
// of a dynamics step finds it (test_lmdamc.f:mcload).
double mcLoad()
{
   REQUIRE(use_lmdamc);
   REQUIRE(lmdamctrial == 1);
   if (use_abf) {
      // give the mean force a slope
      for (int i = 1; i <= nlmda; ++i)
         lmdafmean[i] = 30.0 * (double)(i - 1) * wlmda;
   } else if (use_ost) {
      energy(calc::v4);
      hbias = 2.0;
      lmdaavg = lambda;
      dedlavg = dedl;
      ostDeposit(1);
   } else {
      REQUIRE(use_meta);
      hbias = 2.0;
      lmdaavg = lambda;
      metaDeposit(1);
   }
   energy(calc::v4);
   REQUIRE(esum_epoch == xyz_epoch);
   return esum;
}

// The energy call a lambda move makes for the method in use.
int trialVers()
{
   return use_ost ? calc::v4 : calc::energy;
}

// The potential energy plus the adaptive bias at the current lambda, found
// independently of the dynamics step, with the solve of a trial
// (test_lmdamc.f:mcenergy).
double mcEnergy()
{
   InducedTrial trial(InducedTrial::History::BYPASS);
   const energy_prec e0 = esum;
   const auto epoch0 = esum_epoch;
   const LmdaStepState step;
   const auto gx0 = grab1(gx), gy0 = grab1(gy), gz0 = grab1(gz);
   nblistRefresh();
   energy(trialVers());
   const double e = esum;
   // leave the step as it was, but for the dipoles the caller compares
   darray::copyin(g::q0, n, gx, gx0.data());
   darray::copyin(g::q0, n, gy, gy0.data());
   darray::copyin(g::q0, n, gz, gz0.data());
   waitFor(g::q0);
   step.restore();
   esum = e0;
   esum_epoch = epoch0;
   trial.accept();
   return e;
}

// Tries one lambda move from the lambda \c lmda on step \c istep, handing over
// \c efake as the energy of the step, and reports whether the move was
// accepted (test_lmdamc.f:mctrial).
bool mcTrial(double lmda, int istep, double efake)
{
   lambda = lmda;
   esum = efake;
   const int nacc = nlmdamcacc;
   lmdaMonte(istep);
   return nlmdamcacc > nacc;
}

// What a scaling barostat leaves behind: the box and the coordinates set after
// the force call of the step, with the neighbor lists not yet refreshed.
void rescaleBox(double scale)
{
   lvec1 *= scale;
   lvec2 *= scale;
   lvec3 *= scale;
   boxSetCurrentRecip();
   pos_prec* dev[] = {xpos, ypos, zpos};
   for (auto* d : dev) {
      auto v = grab1(d);
      for (auto& c : v)
         c *= scale;
      darray::copyin(g::q0, n, d, v.data());
   }
   waitFor(g::q0);
   copyPosToXyz();
}

// Forces a rejected and then an accepted lambda move for one method, with the
// flags \c rc (test_lmdamc.f:test_lmdamc_move).
void runMove(const char* mode, int rc)
{
   const double eps = testGetEps(1.0e-5, 1.0e-6);
   const double eps_u = testGetEps(1.0e-5, 1.0e-8);
   double lnew = 0, unew = 0;

   {
      Setup s(kIonwat, mode, rc);
      REQUIRE(use(Potent::POLAR));
      const double e0 = mcLoad();
      const double lold = lambda;
      const State before;
      const auto g0 = gradient();

      // an old energy far below the trial energy rejects the move
      esum = -1.0e30;
      lmdaMonte(1);
      REQUIRE(nlmdamctry == 1);
      REQUIRE(nlmdamcacc == 0);
      REQUIRE(esum == -1.0e30);
      REQUIRE(esum_epoch == xyz_epoch);
      requireSame(State(), before);

      // The lambda dependent parameters are scaled again by the next energy
      // call, which finds the energy and forces of the step once more.
      energy(calc::v4);
      COMPARE_REALS(esum, e0, eps);
      REQUIRE(maxDiff(gradient(), g0) < testGetEps(1.0e-4, 1.0e-6));
      const State step;

      // An old energy far above the trial energy accepts the move, which is
      // kept within the gaussian of the bias so that the bias at the new lambda
      // is not zero.
      lmdamcmove = 2.0 * wlmda;
      lmdavtheta = 1.0;
      esum = e0 + 1.0e6;
      lmdaMonte(1);
      REQUIRE(nlmdamctry == 2);
      REQUIRE(nlmdamcacc == 1);
      REQUIRE(lambda != lold);
      REQUIRE(std::fabs(lambda - lold) <= lmdamcmove);
      REQUIRE(esum_epoch == xyz_epoch);
      // the virial and the forces of the step stay in place
      REQUIRE(sameBits(std::vector<virial_prec>(vir, vir + 9), step.vir));
      REQUIRE(sameBits(grab1(gx), step.gx));
      REQUIRE(sameBits(grab1(gy), step.gy));
      REQUIRE(sameBits(grab1(gz), step.gz));
      // the lambda particle sits at the new lambda, at rest
      double lmap, dldth;
      lmdaThetaMap(lmdatheta, lmap, dldth);
      COMPARE_REALS(lmap, lambda, 1.0e-10);
      REQUIRE(lmdavtheta == 0.0);
      REQUIRE(nualt == step.nualt);
      // the sublambdas and the bias scalars are those of the new lambda
      const double el = elam, pl = plam, vl = vlam;
      mapSubLambda();
      REQUIRE(elam == el);
      REQUIRE(plam == pl);
      REQUIRE(vlam == vl);
      REQUIRE(biasInEsum() != 0.0);
      if (use_ost or use_abf)
         REQUIRE(lmdaddgdl == lmdadfdl);
      if (use_ost) {
         // the bias follows the lambda derivative found at the new lambda
         REQUIRE(dedl != step.dl[0]);
         REQUIRE(ostdgdl != 0.0);
         REQUIRE(ostdgdl == bdgdl + bdgdfl * d2edl2);
      } else {
         // an energy alone is found, which leaves the lambda derivative of the step
         REQUIRE(dedl == step.dl[0]);
      }

      // the accepted energy and induced dipoles are those of an independent
      // evaluation at the new lambda
      const double eacc = esum;
      const energy_prec dedlacc = dedl;
      const auto uacc = grab3(uind), pacc = grab3(uinp);
      const double eref = mcEnergy();
      COMPARE_REALS(eacc, eref, eps);
      REQUIRE(maxDiff(grab3(uind), uacc) < eps_u);
      REQUIRE(maxDiff(grab3(uinp), pacc) < eps_u);
      REQUIRE(dedl == dedlacc);

      lnew = lambda;
      unew = eacc - biasInEsum();
      s.session.end();
   }

   // The same structure set up afresh at the new lambda, with no bias yet.
   {
      Setup s(kIonwat, mode, kRc);
      lambda = lnew;
      energy(calc::energy);
      REQUIRE(biasInEsum() == 0.0);
      COMPARE_REALS(unew, esum, eps);
      s.session.end();
   }
}

// The old energy of a lambda move is found again when the energy of the step
// no longer describes the system, and trusted otherwise. A large constant mean
// force makes the energy with its bias rise steeply with lambda, so that at a
// very low temperature every move from a lambda of one is downhill and
// accepted, and every move from a lambda of zero is uphill and rejected, as
// long as the old energy is the true one; a false energy handed over for the
// step shows whether it was used (test_lmdamc.f:test_lmdamc_stale).
void runStale(const char* mode)
{
   const double eps = testGetEps(1.0e-5, 1.0e-6);
   Setup s(kIonwat, mode, kRcDyn);
   mcLoad();
   const bool ost = use_ost;
   for (int i = 1; i <= nlmda; ++i)
      lmdafmean[i] = -1.0e5;
   bath::kelvin = 1.0e-6;

   // After a scaling barostat a downhill move is accepted whatever energy the
   // step hands over, and it returns the true new energy.
   rescaleBox(1.02);
   REQUIRE(esum_epoch != xyz_epoch);
   REQUIRE(mcTrial(1.0, 1, -1.0e30));
   REQUIRE(lambda < 1.0);
   REQUIRE(esum_epoch == xyz_epoch);
   const double eacc = esum;
   COMPARE_REALS(eacc, mcEnergy(), eps);

   // An uphill move is rejected, and the rejection leaves the energy of the
   // step and the state that the step left untouched.
   lambda = 0.0;
   mapSubLambda();
   copyPosToXyz();
   const State before;
   REQUIRE_FALSE(mcTrial(0.0, 1, 1.0e30));
   REQUIRE(esum == 1.0e30);
   REQUIRE(esum_epoch != xyz_epoch);
   requireSame(State(), before);

   // With nothing moved the energy of the step is trusted, so the same false
   // energies now decide both moves the other way.
   energy(calc::v4);
   REQUIRE_FALSE(mcTrial(1.0, 1, -1.0e30));
   REQUIRE(mcTrial(0.0, 1, 1.0e30));

   // On a step that deposits, orthogonal space tempering finds its energy
   // again, while adaptive biasing force only has to take its bias from the
   // history as it now stands.
   energy(calc::v4);
   lmdanpa = lmdaintv;
   REQUIRE(mcTrial(1.0, lmdaintv, -1.0e30) == ost);

   s.session.end();
}
}

TEST_CASE("LMDAMC-keys", "[ff][lmdamc]")
{
   // an adaptive bias run moves the lambda particle by default
   {
      Setup s("mutate/179_ionwat_ast_l05.key", "abf", kRc);
      REQUIRE(use_abf);
      REQUIRE_FALSE(use_lmdamc);
      REQUIRE(lmdaintv == 100);
      REQUIRE(lmdanpa == 30);
      REQUIRE(lmdanpb == 30);
      REQUIRE(lmdanpc == 40);
      REQUIRE(lmdamctrial == 10);
      REQUIRE(lmdamcmove == 0.02);
      s.session.end();
   }

   // the keywords turn on the moves and set their rate and size
   {
      Setup s(kIonwat, "abf", kRc);
      REQUIRE(use_lmdamc);
      REQUIRE(lmdaintv == 4);
      REQUIRE(lmdanpa == 2);
      REQUIRE(lmdamctrial == 1);
      REQUIRE(lmdamcmove == 0.05);
      REQUIRE(nlmdamctry == 0);
      REQUIRE(nlmdamcacc == 0);
      s.session.end();
   }

   // a move larger than the lambda range is cut back to the range
   {
      Setup s(kIonwat, "abf", kRc, 0., "lambda-mc-move -3.0\n");
      REQUIRE(lmdamcmove == 1.0);
      s.session.end();
   }
}

TEST_CASE("LMDAMC-particle", "[ff][lmdamc]")
{
   // The interval sampling no longer propagates the lambda particle once the
   // lambda is moved by Monte Carlo, while the samples are still deposited
   // (test_lmdamc.f:test_lmdamc_particle).
   Setup s(kIonwat, "abf", kRc);
   lambda = 0.5;
   lmdaThetaInv(lambda, lmdatheta);
   const double theta0 = lmdatheta;
   lmdavtheta = 0.0;
   lmdadfdl = 0.0;
   int istep = 0;
   for (int i = 0; i < 2 * lmdaintv; ++i) {
      dedl = 40.0;
      elmdaDyn(++istep);
   }
   REQUIRE(lambda == 0.5);
   REQUIRE(lmdatheta == theta0);
   REQUIRE(lmdavtheta == 0.0);
   REQUIRE(nlmdahist == 2);

   // the same steps move the lambda once the particle is in charge
   use_lmdamc = false;
   for (int i = 0; i < lmdaintv; ++i) {
      dedl = 40.0;
      elmdaDyn(++istep);
   }
   REQUIRE(lambda != 0.5);
   s.session.end();
}

TEST_CASE("LMDAMC-move-abf", "[ff][lmdamc]") { runMove("abf", kRc); }
TEST_CASE("LMDAMC-move-ost", "[ff][lmdamc]") { runMove("ost", kRc); }
TEST_CASE("LMDAMC-move-meta", "[ff][lmdamc]") { runMove("meta", kRc); }
TEST_CASE("LMDAMC-move-dyn-abf", "[ff][lmdamc]") { runMove("abf", kRcDyn); }
TEST_CASE("LMDAMC-move-dyn-ost", "[ff][lmdamc]") { runMove("ost", kRcDyn); }
TEST_CASE("LMDAMC-move-dyn-meta", "[ff][lmdamc]") { runMove("meta", kRcDyn); }

TEST_CASE("LMDAMC-trial", "[ff][lmdamc]")
{
   // Orthogonal space tempering and adaptive biasing force try a lambda move
   // only in the propagation phase of the sample interval while metadynamics
   // tries at any step, the mean trial rate is honored, and a move from either
   // end of the lambda range is reflected back into the range
   // (test_lmdamc.f:test_lmdamc_trial).
   {
      Setup s(kIonwat, "abf", kRcDyn);
      const double e0 = mcLoad();

      // no trial is made once the propagation phase is over
      for (int i = 0; i < 10; ++i) {
         for (int istep : {3, 4, 7, 8}) {
            esum = e0 + 1.0e6;
            lmdaMonte(istep);
         }
      }
      REQUIRE(nlmdamctry == 0);
      REQUIRE(lambda == 0.5);
      REQUIRE(esum == e0 + 1.0e6);

      // a long mean time between trials leaves the lambda alone
      lmdamctrial = 1000000000;
      for (int i = 0; i < 10; ++i) {
         esum = e0 + 1.0e6;
         lmdaMonte(1);
      }
      REQUIRE(nlmdamctry == 0);
      lmdamctrial = 1;

      // both steps of the propagation phase try, in every interval
      for (int istep : {1, 2, 5, 6})
         mcTrial(0.5, istep, -1.0e30);
      REQUIRE(nlmdamctry == 4);
      REQUIRE(nlmdamcacc == 0);

      // moves from the lambda end points are reflected into the range
      double lmin = 1.0, lmax = 0.0;
      for (int i = 0; i < 10; ++i) {
         REQUIRE(mcTrial(0.0, 1, 1.0e30));
         lmin = std::min(lmin, lambda);
         lmax = std::max(lmax, lambda);
      }
      REQUIRE(lmin >= 0.0);
      REQUIRE(lmax <= lmdamcmove);
      lmin = 1.0, lmax = 0.0;
      for (int i = 0; i < 10; ++i) {
         REQUIRE(mcTrial(1.0, 1, 1.0e30));
         lmin = std::min(lmin, lambda);
         lmax = std::max(lmax, lambda);
      }
      REQUIRE(lmax <= 1.0);
      REQUIRE(lmin >= 1.0 - lmdamcmove);
      REQUIRE(nlmdamcacc == 20);
      s.session.end();
   }

   // metadynamics tries a move in every phase of the interval
   {
      Setup s(kIonwat, "meta", kRcDyn);
      mcLoad();
      for (int i = 0; i < 5; ++i) {
         REQUIRE_FALSE(mcTrial(0.5, 3, -1.0e30));
         REQUIRE_FALSE(mcTrial(0.5, 4, -1.0e30));
      }
      REQUIRE(nlmdamctry == 10);
      REQUIRE(nlmdamcacc == 0);
      s.session.end();
   }
}

TEST_CASE("LMDAMC-deposit", "[ff][lmdamc]")
{
   // A lambda move tried on the step that deposits a metadynamics gaussian
   // takes its old bias from the history with that gaussian in place; the
   // energy of the step was found before the deposit, and a move judged
   // against it would see the whole of the new gaussian as an energy barrier
   // (test_lmdamc.f:test_lmdamc_deposit).
   Setup s(kIonwat, "meta", kRcDyn);
   const double e0 = mcLoad();

   // a tall gaussian lands on the current lambda after the forces of the step
   // were found, as at the end of a sample interval
   hbias = 1000.0;
   lmdaavg = lambda;
   metaDeposit(4);

   // a move of up to two lambda bins leaves the peak, so it is downhill
   lmdamcmove = 2.0 * wlmda;
   const double lold = lambda;
   REQUIRE(mcTrial(lold, 4, e0));
   REQUIRE(nlmdamctry == 1);
   REQUIRE(lambda != lold);
   const double eacc = esum;
   COMPARE_REALS(eacc, mcEnergy(), testGetEps(1.0e-5, 1.0e-6));
   s.session.end();
}

TEST_CASE("LMDAMC-bias-abf", "[ff][lmdamc]")
{
   // The old energy of an adaptive biasing force move is the true energy of the
   // step with the bias of the step taken out and that of the history as it
   // now stands put in. Both parts are checked with the true energy handed
   // over, at a temperature so low that a move is accepted only if downhill.
   const double kelvin = 1.0e-6;

   // The mean force falls steeply over most of the lambda range and rises over
   // the last tenth, so the free energy at a lambda of one is far below zero
   // while every move down from one is uphill in the bias. Leaving the bias of
   // the step inside the old energy would make each of them look downhill.
   {
      Setup s(kIonwat, "abf", kRcDyn);
      for (int i = 1; i <= nlmda; ++i)
         lmdafmean[i] = ((double)(i - 1) * wlmda < 0.9) ? -1.0e5 : 1.0e5;
      bath::kelvin = kelvin;
      for (int i = 0; i < 10; ++i) {
         lambda = 1.0;
         energy(calc::v4);
         REQUIRE(bostlmda < -1.0e4);
         REQUIRE_FALSE(mcTrial(1.0, 1, esum));
      }
      REQUIRE(nlmdamctry == 10);
      s.session.end();
   }

   // The mean force changes after the forces of the step were found, as a
   // deposit at the end of a sample interval changes it. With the new mean
   // force every move down from a lambda of one is uphill; judged with the
   // bias of the step, which was zero, each would be downhill by the whole of
   // the new free energy.
   {
      Setup s(kIonwat, "abf", kRcDyn);
      bath::kelvin = kelvin;
      for (int i = 0; i < 10; ++i) {
         for (int j = 1; j <= nlmda; ++j)
            lmdafmean[j] = 0.0;
         lambda = 1.0;
         energy(calc::v4);
         REQUIRE(bostlmda == 0.0);
         for (int j = 1; j <= nlmda; ++j)
            lmdafmean[j] = 1.0e5;
         REQUIRE_FALSE(mcTrial(1.0, 1, esum));
      }
      REQUIRE(nlmdamctry == 10);
      s.session.end();
   }
}

TEST_CASE("LMDAMC-stale-abf", "[ff][lmdamc]") { runStale("abf"); }
TEST_CASE("LMDAMC-stale-ost", "[ff][lmdamc]") { runStale("ost"); }

TEST_CASE("LMDAMC-stale-meta", "[ff][lmdamc]")
{
   // The same check for metadynamics, where a tall gaussian centered on the
   // current lambda makes every nearby move downhill once the old energy is
   // the true one (test_lmdamc.f:test_lmdamc_stalemeta).
   Setup s(kIonwat, "meta", kRcDyn);
   mcLoad();
   const double lpeak = lambda;
   hbias = 1000.0;
   lmdaavg = lpeak;
   metaDeposit(4);
   // a move of up to two lambda bins leaves the peak, so it is downhill
   lmdamcmove = 2.0 * wlmda;

   // after a scaling barostat the old energy is found again
   rescaleBox(1.02);
   REQUIRE(mcTrial(lpeak, 3, -1.0e30));
   const double eacc = esum;
   COMPARE_REALS(eacc, mcEnergy(), testGetEps(1.0e-5, 1.0e-6));

   // with nothing moved the energy of the step is trusted
   energy(calc::v4);
   REQUIRE_FALSE(mcTrial(lpeak, 3, -1.0e30));
   s.session.end();
}

TEST_CASE("LMDAMC-stale-lists", "[ff][lmdamc]")
{
   // A box that a scaling barostat has shrunk by more than the buffer of the
   // neighbor lists brings pairs inside the cutoffs that the lists of the step
   // do not hold, so the lists are refreshed before the energies of the move
   // are found. Ten waters with a thin buffer; the reference evaluation
   // refreshes the lists itself.
   const std::string mc = "lambda-propagate mc\nlambda-mc-trial 1\nlambda-mc-move 0.05\nlambda-interval 4\n"
                          "lambda-eqratio 0.5 0.25\nlist-buffer 0.5\n";
   Setup s("mutate/128_water_ast_ye_l05.key", "abf", kRcDyn, 0., mc, "mutate/water2.xyz",
      "commit_6fe8e913/water03.prm");
   REQUIRE(n == 30);
   for (int i = 1; i <= nlmda; ++i)
      lmdafmean[i] = -1.0e5;
   bath::kelvin = 1.0e-6;
   energy(calc::v4);

   rescaleBox(0.9);
   REQUIRE(mcTrial(1.0, 1, -1.0e30));
   const double eacc = esum;
   COMPARE_REALS(eacc, mcEnergy(), testGetEps(1.0e-5, 1.0e-6));
   s.session.end();
}

TEST_CASE("LMDAMC-after-volume-move", "[ff][lmdamc]")
{
   // A Monte Carlo box change in the step leaves an energy that matches the
   // system it leaves, whether it was accepted or rejected, so the lambda move
   // that follows trusts it.
   Setup s(kIonwat, "abf", kRcDyn, 1.);
   mcLoad();
   for (int i = 1; i <= nlmda; ++i)
      lmdafmean[i] = -1.0e5;
   const int voltrial0 = bath::voltrial;
   const double volmove0 = bath::volmove;
   FstrView volscale = bath::volscale;
   const std::string volscale0 = volscale.trim();
   bath::voltrial = 1;
   bath::volmove = 200.;
   volscale = "MOLECULAR";

   // the barostat owns device arrays, so it has to go before the session ends
   {
      MonteCarloBarostat mc;
      Box box0, box1;

      // an old energy far below any trial energy rejects the volume move
      energy(calc::v4);
      boxGetCurrent(box0);
      monteCarloBarostat(-1.0e30, 298., false, false);
      boxGetCurrent(box1);
      REQUIRE(box1.lvec1.x == box0.lvec1.x);
      REQUIRE(esum == -1.0e30);
      REQUIRE(esum_epoch == xyz_epoch);
      bath::kelvin = 1.0e-6;
      REQUIRE_FALSE(mcTrial(1.0, 1, -1.0e30));

      // an old energy far above the trial energy accepts it
      bath::kelvin = 298.;
      lambda = 0.5;
      energy(calc::v4);
      monteCarloBarostat(esum + 1.0e6, 298., false, false);
      boxGetCurrent(box1);
      REQUIRE(box1.lvec1.x != box0.lvec1.x);
      REQUIRE(esum_epoch == xyz_epoch);
      bath::kelvin = 1.0e-6;
      REQUIRE_FALSE(mcTrial(1.0, 1, -1.0e30));
      REQUIRE(mcTrial(0.0, 1, 1.0e30));
   }

   bath::voltrial = voltrial0;
   bath::volmove = volmove0;
   volscale = volscale0;
   s.session.end();
}

TEST_CASE("LMDAMC-rpole", "[ff][lmdamc]")
{
   // The saved output reads the multipoles in the global frame after asking
   // for the physical ones (mdsaveAsync). A rejected lambda move leaves them
   // as the step left them, with and without polarization, and an accepted
   // one leaves those of the new lambda (test_lmdamc.f:test_lmdamc_nopol).
   for (const char* key : {kNopol, kIonwat}) {
      CAPTURE(key);
      Setup s(key, "abf", kRcDyn);
      const double e0 = mcLoad();
      mpoleEnsurePhysical();
      const auto rpole0 = grabRpole();

      REQUIRE_FALSE(mcTrial(lambda, 1, -1.0e30));
      mpoleEnsurePhysical();
      REQUIRE(sameBits(grabRpole(), rpole0));

      REQUIRE(mcTrial(lambda, 1, e0 + 1.0e6));
      mpoleEnsurePhysical();
      REQUIRE_FALSE(sameBits(grabRpole(), rpole0));
      s.session.end();
   }
}

TEST_CASE("LMDAMC-predictor", "[ff][lmdamc]")
{
   // The dynamics of an adaptive bias method keeps the induced dipole
   // predictor that the key file asks for, which the Fortran reader turns off,
   // and a lambda move neither reads nor writes its history.
   for (const char* mode : {"abf", "meta"}) {
      CAPTURE(mode);
      Setup s(kIonwat, mode, kMd, 0., "integrator verlet\n");
      REQUIRE(polpred == UPred::ASPC);
      REQUIRE(maxualt == 16);
      const double e0 = mcLoad();
      for (int i = 0; i < 3; ++i)
         energy(calc::v4);
      REQUIRE(nualt >= 3);
      const State before;

      REQUIRE_FALSE(mcTrial(lambda, 1, -1.0e30));
      requireSame(State(), before);

      REQUIRE(mcTrial(lambda, 1, e0 + 1.0e6));
      const State after;
      REQUIRE(after.nualt == before.nualt);
      for (size_t i = 0; i < before.slots.size(); ++i) {
         REQUIRE(sameBits(after.slots[i], before.slots[i]));
         REQUIRE(sameBits(after.pslots[i], before.pslots[i]));
      }
      s.session.end();
   }

   // Dual topology polarization, which orthogonal space tempering uses with a
   // polarization map, keeps a predictor history for each endpoint, and a
   // lambda move reads and writes neither.
   {
      Setup s(kIonwat, "ost", kMd, 0., "integrator verlet\n");
      REQUIRE(use_epdt);
      REQUIRE(polpred == UPred::ASPC);
      REQUIRE(induceHistoryUsable());
      const double e0 = mcLoad();
      for (int i = 0; i < 3; ++i)
         energy(calc::v4);
      DtPass pass[2];
      dtPasses(pass);
      std::vector<State> before;
      for (int k : {0, 1}) {
         induceLoadHistory(pass[k].mask);
         before.emplace_back();
         REQUIRE(before.back().nualt >= 3);
      }
      // the two endpoints solve different subsystems
      const int last = (before[0].nualt - 1) % maxualt;
      REQUIRE_FALSE(sameBits(before[0].slots[last], before[1].slots[last]));

      REQUIRE_FALSE(mcTrial(lambda, 1, -1.0e30));
      REQUIRE(mcTrial(lambda, 1, e0 + 1.0e6));
      for (int k : {0, 1}) {
         CAPTURE(k);
         induceLoadHistory(pass[k].mask);
         const State after;
         REQUIRE(after.nualt == before[k].nualt);
         for (size_t i = 0; i < before[k].slots.size(); ++i) {
            REQUIRE(sameBits(after.slots[i], before[k].slots[i]));
            REQUIRE(sameBits(after.pslots[i], before[k].pslots[i]));
         }
      }
      s.session.end();
   }
}

namespace {
// the history length one more solve leaves (induce.cpp:ulspredSave)
int nextNualt(int nu)
{
   ++nu;
   return nu > 2 * maxualt ? nu - maxualt : nu;
}
}

TEST_CASE("LMDAMC-predictor-restart", "[ff][lmdamc]")
{
   // A single topology history holds the dipoles of one polarization lambda.
   // A rejected move leaves the lambda where it was and the predictor carries
   // on; an accepted one moves it, and the next solve starts over from the
   // direct guess.
   for (const char* mode : {"abf", "meta"}) {
      CAPTURE(mode);
      Setup s(kIonwat, mode, kMd, 0., "integrator verlet\n");
      REQUIRE_FALSE(use_epdt);
      REQUIRE(use_prst);
      const double e0 = mcLoad();
      for (int i = 0; i < maxualt; ++i)
         energy(calc::v4);
      REQUIRE(induceGuess() == UGuess::PREDICT);

      const double l0 = lambda;
      int nu = nualt;
      REQUIRE_FALSE(mcTrial(l0, 1, -1.0e30));
      REQUIRE(lambda == l0);
      energy(calc::v4);
      REQUIRE(nualt == nextNualt(nu));
      REQUIRE(induceGuess() == UGuess::PREDICT);

      REQUIRE(mcTrial(l0, 1, e0 + 1.0e6));
      REQUIRE(lambda != l0);
      energy(calc::v4);
      REQUIRE(nualt == 1);
      REQUIRE(induceGuess() == UGuess::DIRECT);
      s.session.end();
   }
}

TEST_CASE("LMDAMC-predictor-phases", "[ff][lmdamc]")
{
   // ABF moves a single topology lambda only in the propagation phase of each
   // interval and holds it through the equilibration and averaging phases, so
   // the predictor history starts over with each solve while lambda moves and
   // fills up while it is held.
   Setup s(kIonwat, "abf", kMd, 0.,
      "integrator verlet\nlambda-propagate langevin\nlambda-interval 40\nlambda-eqratio 0.25 0.25\n");
   REQUIRE_FALSE(use_lmdamc);
   REQUIRE_FALSE(use_epdt);
   REQUIRE(lmdanpa == 10);
   REQUIRE(lmdanpa + lmdanpb + lmdanpc == 40);
   TestRemoveFileOnExit fdyn("ionwat.dyn");
   const int oldiwrite = inform::iwrite;
   inform::iwrite = 1000;

   // lambda moves after each of the first 10 steps, so the force calls of the
   // next 20 are the ones at the held lambda
   const double l0 = lambda;
   mdPropagate(30, 0.001);
   REQUIRE(lambda != l0);
   REQUIRE(nualt == 20);
   REQUIRE(induceGuess() == UGuess::PREDICT);

   inform::iwrite = oldiwrite;
   s.session.end();
}

TEST_CASE("LMDAMC-thermostat", "[ff][lmdamc]")
{
   // Monte Carlo lambda moves are accepted at a set temperature, so a run
   // without a thermostat is refused (dynamic.f).
   Setup s(kIonwat, "abf", kRc);
   REQUIRE(dlmda::use_lmdamc);
   REQUIRE_NOTHROW(lmdaMonteCheck());
   bath::isothermal = 0;
   REQUIRE_THROWS(lmdaMonteCheck());
   dlmda::use_lmdamc = 0;
   REQUIRE_NOTHROW(lmdaMonteCheck());
   s.session.end();
}

TEST_CASE("LMDAMC-run", "[ff][lmdamc]")
{
   // Short dynamics runs with Monte Carlo lambda moves for each adaptive bias
   // method, at constant volume and under the Monte Carlo and the scaling
   // barostats (test_lmdamc.f:test_lmdamc_run).
   struct Run
   {
      const char* mode;
      const char* baro; // empty at constant volume
   };
   const Run runs[] = {{"abf", ""}, {"meta", ""}, {"ost", ""}, {"ost", "montecarlo"}, {"abf", "bussi"},
      {"meta", "bussi"}, {"ost", "berendsen"}};
   const int nstep = 60;

   for (const auto& r : runs) {
      CAPTURE(r.mode, r.baro);
      const bool npt = r.baro[0] != 0;
      std::string extra = "integrator verlet\nthermostat bussi\nlambda-mc-trial 2\nrandomseed 123456789\n";
      if (npt)
         extra += std::string("barostat ") + r.baro + "\nvolume-trial 3\n";
      Setup s(kIonwat, r.mode, kMd, npt ? 1. : 0., extra);
      // mdPropagate() saves the restart file of the run as it returns
      TestRemoveFileOnExit fdyn("ionwat.dyn");
      const int oldiwrite = inform::iwrite;
      inform::iwrite = 1000;

      mdPropagate(nstep, 0.001);

      REQUIRE(std::isfinite(esum));
      REQUIRE(lambda >= 0.0);
      REQUIRE(lambda <= 1.0);
      REQUIRE(nlmdamctry > 0);
      REQUIRE(nlmdamcacc <= nlmdamctry);
      // the lambda particle follows the accepted moves
      double lmap, dldth;
      lmdaThetaMap(lmdatheta, lmap, dldth);
      if (nlmdamcacc > 0)
         COMPARE_REALS(lmap, lambda, 1.0e-10);
      // the samples are still deposited
      if (use_meta)
         REQUIRE(nmetahist == nstep / lmdaintv);
      else
         REQUIRE(nlmdahist == nstep / lmdaintv);

      inform::iwrite = oldiwrite;
      s.session.end();
   }
}

TEST_CASE("LMDAMC-stale-table", "[ff][lmdamc]")
{
   // Which integrators and barostats leave the energy of the step behind the
   // system: only the barostats that rescale the box and the coordinates after
   // the force call. Everything else moves the atoms and the box before it.
   // The leapfrog Langevin piston integrator has no CUDA kernels (mdVelB), so
   // it cannot run lambda dynamics and is left out.
   struct Step
   {
      const char* name;
      const char* keys;
      double atm;
      std::function<BasicIntegrator*()> make;
      bool stale;
   };
   auto verlet = [](ThermostatEnum t, BarostatEnum b) { return [=] { return (BasicIntegrator*)new VerletIntegrator(t, b); }; };
   auto respa = [](ThermostatEnum t, BarostatEnum b) { return [=] { return (BasicIntegrator*)new RespaIntegrator(t, b); }; };
   const ThermostatEnum bussi = ThermostatEnum::BUSSI;
   const Step steps[] = {
      {"verlet nvt", "", 0., verlet(bussi, BarostatEnum::NONE), false},
      {"verlet nvt rattle", "rattle water\n", 0., verlet(bussi, BarostatEnum::NONE), false},
      {"verlet montecarlo", "", 1., verlet(bussi, BarostatEnum::MONTECARLO), false},
      {"verlet berendsen", "", 1., verlet(bussi, BarostatEnum::BERENDSEN), true},
      {"verlet bussi", "", 1., verlet(bussi, BarostatEnum::BUSSI), true},
      {"verlet bussi rattle", "rattle water\n", 1., verlet(bussi, BarostatEnum::BUSSI), true},
      {"respa nvt", "respa-inner 4\n", 0., respa(bussi, BarostatEnum::NONE), false},
      {"respa montecarlo", "respa-inner 4\n", 1., respa(bussi, BarostatEnum::MONTECARLO), false},
      {"respa berendsen", "respa-inner 4\n", 1., respa(ThermostatEnum::NHC, BarostatEnum::BERENDSEN), true},
      {"nhc1996", "", 1., [] { return (BasicIntegrator*)new Nhc96Integrator; }, false},
      {"nhc2006", "", 1., [] { return (BasicIntegrator*)new Nhc06Integrator(1); }, false},
      {"nhc2006 rattle", "rattle water\n", 1., [] { return (BasicIntegrator*)new Nhc06Integrator(1); }, false},
      {"lp2022", "", 1., [] { return (BasicIntegrator*)new LP22Integrator(1); }, false},
      {"lp2022 rattle", "rattle water\n", 1., [] { return (BasicIntegrator*)new LP22Integrator(1); }, false},
      {"stochastic", "", 0., [] { return (BasicIntegrator*)new StochasticIntegrator; }, false},
   };

   for (const auto& st : steps) {
      CAPTURE(st.name);
      BathReset bathReset;
      const char* k = "test_water30.key";
      const char* d = "test_water30.dyn";
      const char* x = "test_water30.xyz";
      TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, std::string("\nintegrator verlet\n") + st.keys);
      TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_iso.dyn", d);
      TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x);
      TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
      const char* argv[] = {"dummy", x};
      TestSession session(2, argv);
      testMdInit(298., st.atm);
      rc_flag = kMd;
      session.init();

      // every step finds the energy, and every step tries a volume move
      const int oldiwrite = inform::iwrite, oldvoltrial = bath::voltrial;
      inform::iwrite = 1;
      bath::voltrial = 1;
      {
         std::unique_ptr<BasicIntegrator> intg(st.make());
         for (int istep = 1; istep <= 3; ++istep) {
            intg->dynamic(istep, 0.001);
            CAPTURE(istep);
            REQUIRE((esum_epoch != xyz_epoch) == st.stale);
         }
      }
      inform::iwrite = oldiwrite;
      bath::voltrial = oldvoltrial;
      session.end();
   }
}

#endif
