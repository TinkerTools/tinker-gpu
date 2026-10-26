#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "tool/darray.h"
#include "tool/error.h"

#include "test.h"
#include "testrt.h"

#include <tinker/detail/mplpot.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/potent.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Tests of what an induced dipole solve may touch (src/inducewiggle.cpp): its
// starting guess, what becomes of a solve that runs out of iterations, and a
// trial that puts the induced dipoles and the predictor history back.

using namespace tinker;

namespace {
// induceGuess() reads these, and Catch2 randomizes the case order.
struct GuessScope
{
   UPred pred;
   int nu, maxu, guess, polar;
   bool epdt;

   GuessScope()
      : pred(polpred)
      , nu(nualt)
      , maxu(maxualt)
      , guess(polpcg::pcgguess)
      , polar(potent::use_polar)
      , epdt(use_epdt)
   {}

   ~GuessScope()
   {
      polpred = pred;
      nualt = nu;
      maxualt = maxu;
      polpcg::pcgguess = guess;
      potent::use_polar = polar;
      use_epdt = epdt;
   }
};

void setGuess(UPred pred, bool full, bool epdt, bool pcgguess)
{
   polpred = pred;
   maxualt = 16;
   nualt = full ? 16 : 15;
   use_epdt = epdt;
   polpcg::pcgguess = pcgguess;
}
}

TEST_CASE("INDUCE-guess-table", "[ff][induce]")
{
   GuessScope scope;
   // No system is loaded, so a trial here must find no dipoles to save.
   potent::use_polar = 0;
   REQUIRE_FALSE(induceHistoryBypassed());

   // the predictor, once its history is full
   for (bool pcgguess : {false, true}) {
      setGuess(UPred::ASPC, true, false, pcgguess);
      REQUIRE(induceGuess() == UGuess::PREDICT);
      setGuess(UPred::GEAR, true, false, pcgguess);
      REQUIRE(induceGuess() == UGuess::PREDICT);
   }

   // a predictor that has to be passed over falls back on the direct guess,
   // whatever PCG-GUESS says
   setGuess(UPred::ASPC, false, false, false);
   REQUIRE(induceGuess() == UGuess::DIRECT);
   nualt = 0;
   REQUIRE(induceGuess() == UGuess::DIRECT);

   // dual topology never predicts, and no predictor leaves PCG-GUESS to choose
   setGuess(UPred::ASPC, true, true, true);
   REQUIRE(induceGuess() == UGuess::DIRECT);
   setGuess(UPred::ASPC, true, true, false);
   REQUIRE(induceGuess() == UGuess::ZERO);
   setGuess(UPred::NONE, true, false, true);
   REQUIRE(induceGuess() == UGuess::DIRECT);
   setGuess(UPred::NONE, true, false, false);
   REQUIRE(induceGuess() == UGuess::ZERO);

   // a trial bypasses the history only when asked to, and only while it lasts
   setGuess(UPred::ASPC, true, false, false);
   {
      InducedTrial trial(InducedTrial::History::USE);
      REQUIRE_FALSE(induceHistoryBypassed());
      REQUIRE(induceGuess() == UGuess::PREDICT);
   }
   {
      InducedTrial trial(InducedTrial::History::BYPASS);
      REQUIRE(induceHistoryBypassed());
      REQUIRE(induceGuess() == UGuess::DIRECT);
      // accepting a trial keeps its dipoles, not its bypass
      trial.accept();
      REQUIRE(induceGuess() == UGuess::DIRECT);
   }
   REQUIRE_FALSE(induceHistoryBypassed());
   REQUIRE(induceGuess() == UGuess::PREDICT);

   // without a predictor a bypassing trial changes nothing
   setGuess(UPred::NONE, true, false, false);
   {
      InducedTrial trial(InducedTrial::History::BYPASS);
      REQUIRE(induceGuess() == UGuess::ZERO);
   }
}

namespace {
const int kMd = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;
const int kNoMd = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;

// A system under test/file, set up from a key file of the free energy
// perturbation tests with the lambda mode left or switched off.
struct System
{
   const char* key;
   const char* xyz;
   const char* prm;
};
// Two AMOEBA water molecules in a small water box, with mutual polarization
// and the ASPC predictor.
const System kWater = {"fep/window.key", "mutate/water2.xyz", "commit_6fe8e913/water03.prm"};
// HIPPO dimethylsulfoxide, which keeps the one set of induced dipoles.
const System kHippo = {"fep/hippo.key", "dmso/dmso.xyz", "hippo/hippo19.prm"};
// A sodium ion and a water molecule with exchange polarization.
const System kExpol = {"expol/expol.key", "expol/Nawater.xyz", "expol/expol.prm"};

struct Setup
{
   TestLmdaFlagReset lmdaReset;
   std::string xyz;
   const char* argv[4];
   // prterr writes the coordinates here before a solver throws
   TestRemoveFileOnExit ferr;
   TestFile fxyz, fkey, fprm;
   TestSession session;

   Setup(int rc, const std::string& keyextra = "", const System& sys = kWater)
      : xyz(testBaseName(sys.xyz))
      , argv{"dummy", xyz.c_str(), "-k", "induce.key"}
      , ferr(xyz.substr(0, xyz.rfind('.')) + ".err")
      , fxyz(std::string(TINKER9_DIRSTR "/test/file/") + sys.xyz)
      , fkey(std::string(TINKER9_DIRSTR "/test/file/") + sys.key, "induce.key",
           "\nlambda-mode none\nintegrator verlet\n" + keyextra)
      , fprm(std::string(TINKER9_DIRSTR "/test/file/") + sys.prm)
      , session(4, argv)
   {
      testMdInit(298., 0.);
      rc_flag = rc;
      session.init();
   }
};

std::vector<pos_prec> grab1(const pos_prec* dev)
{
   std::vector<pos_prec> v(n);
   darray::copyout(g::q0, n, v.data(), dev);
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

// Moves every atom by a hundredth of an Angstrom, differently at each step.
void nudge(int step)
{
   std::vector<pos_prec> xb = grab1(xpos);
   for (int i = 0; i < n; ++i)
      xb[i] += 0.01 * ((i + step) % 3 - 1);
   darray::copyin(g::q0, n, xpos, xb.data());
   waitFor(g::q0);
   copyPosToXyz(true);
}

// Fills the predictor history, as a run of dynamics steps would.
void fillHistory()
{
   REQUIRE(use(Potent::POLAR));
   REQUIRE(polpred == UPred::ASPC);
   REQUIRE(maxualt == 16);
   for (int step = 0; step < 19; ++step) {
      nudge(step);
      energy(calc::v1);
   }
   REQUIRE(nualt >= maxualt);
   REQUIRE(induceGuess() == UGuess::PREDICT);
}

real (**const kUd[])[3] = {&udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06, &udalt_07,
   &udalt_08, &udalt_09, &udalt_10, &udalt_11, &udalt_12, &udalt_13, &udalt_14, &udalt_15};
real (**const kUp[])[3] = {&upalt_00, &upalt_01, &upalt_02, &upalt_03, &upalt_04, &upalt_05, &upalt_06, &upalt_07,
   &upalt_08, &upalt_09, &upalt_10, &upalt_11, &upalt_12, &upalt_13, &upalt_14, &upalt_15};

// The induced dipoles and the whole predictor history.
struct Induced
{
   std::vector<real> uind, uinp, udir, udirp;
   int nualt;
   std::vector<std::vector<real>> slots, pslots;

   Induced()
      : uind(grab3(tinker::uind))
      , uinp(grab3(tinker::uinp))
      , udir(grab3(tinker::udir))
      , udirp(grab3(tinker::udirp))
      , nualt(tinker::nualt)
   {
      for (int i = 0; i < maxualt; ++i) {
         slots.push_back(grab3(*kUd[i]));
         pslots.push_back(grab3(*kUp[i]));
      }
   }
};

bool sameBits(const std::vector<real>& a, const std::vector<real>& b)
{
   return a.size() == b.size() and std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0;
}

bool sameDipoles(const Induced& a, const Induced& b)
{
   return sameBits(a.uind, b.uind) and sameBits(a.uinp, b.uinp) and sameBits(a.udir, b.udir)
      and sameBits(a.udirp, b.udirp);
}

bool sameHistory(const Induced& a, const Induced& b)
{
   if (a.nualt != b.nualt or a.slots.size() != b.slots.size())
      return false;
   for (size_t i = 0; i < a.slots.size(); ++i)
      if (not sameBits(a.slots[i], b.slots[i]) or not sameBits(a.pslots[i], b.pslots[i]))
         return false;
   return true;
}
}

// A wiggle moves the atoms away from the coordinates the history was built at,
// so the retry starts from the direct guess; its solve is saved all the same.
TEST_CASE("INDUCE-retry-guess", "[ff][induce]")
{
   Setup s(kMd, "induce-wiggle\n");
   fillHistory();
   REQUIRE_FALSE(induceFailed());

   // a solver that ran out of iterations reports, and leaves the retry to energy()
   REQUIRE_NOTHROW(induceNotConverged(udir, udirp));
   REQUIRE(induceFailed());

   const auto x0 = grab1(xpos), y0 = grab1(ypos), z0 = grab1(zpos);
   REQUIRE(induceWiggleRetry());
   REQUIRE_FALSE(induceFailed());
   const auto x1 = grab1(xpos), y1 = grab1(ypos), z1 = grab1(zpos);
   // every atom by the first wiggle, 1e-5 Angstrom, in a direction of its own
   double least = 1, most = 0;
   for (int i = 0; i < n; ++i) {
      double dx = x1[i] - x0[i], dy = y1[i] - y0[i], dz = z1[i] - z0[i];
      double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      least = std::min(least, d);
      most = std::max(most, d);
   }
   REQUIRE(least > 0);
   REQUIRE(most < 2.0e-5);

   REQUIRE(induceGuess() == UGuess::DIRECT);
   const int nualt0 = nualt;
   energy(calc::v1);
   REQUIRE(nualt == nualt0 + 1);
   REQUIRE_FALSE(induceFailed());
   REQUIRE(induceGuess() == UGuess::PREDICT);

   s.session.end();
}

TEST_CASE("INDUCE-not-converged-throws", "[ff][induce]")
{
   SECTION("without the keyword")
   {
      Setup s(kMd);
      energy(calc::v1);
      REQUIRE_THROWS_AS(induceNotConverged(udir, udirp), FatalError);
      REQUIRE_FALSE(induceFailed());
      s.session.end();
   }

   SECTION("outside dynamics")
   {
      Setup s(kNoMd, "induce-wiggle\n");
      energy(calc::v1);
      REQUIRE_THROWS_AS(induceNotConverged(udir, udirp), FatalError);
      REQUIRE_FALSE(induceFailed());
      s.session.end();
   }

   SECTION("within a trial that bypasses the history")
   {
      Setup s(kMd, "induce-wiggle\n");
      energy(calc::v1);
      {
         InducedTrial trial(InducedTrial::History::BYPASS);
         REQUIRE_THROWS_AS(induceNotConverged(udir, udirp), FatalError);
         REQUIRE_FALSE(induceFailed());
      }
      // a trial that uses the history is retried as the dynamics is
      {
         InducedTrial trial(InducedTrial::History::USE);
         REQUIRE_NOTHROW(induceNotConverged(udir, udirp));
         REQUIRE(induceFailed());
      }
      s.session.end();
   }
}

TEST_CASE("INDUCE-trial-history", "[ff][induce]")
{
   Setup s(kMd);
   fillHistory();
   const Induced before;

   SECTION("used and rejected")
   {
      {
         InducedTrial trial(InducedTrial::History::USE);
         nudge(19);
         energy(calc::v1);
         const Induced within;
         REQUIRE_FALSE(sameDipoles(within, before));
         REQUIRE(within.nualt == before.nualt + 1);
      }
      const Induced after;
      REQUIRE(sameDipoles(after, before));
      REQUIRE(sameHistory(after, before));
   }

   SECTION("used and accepted")
   {
      Induced within;
      {
         InducedTrial trial(InducedTrial::History::USE);
         nudge(19);
         energy(calc::v1);
         within = Induced();
         trial.accept();
      }
      const Induced after;
      REQUIRE(sameDipoles(after, within));
      REQUIRE(sameHistory(after, within));
      REQUIRE_FALSE(sameDipoles(after, before));
      REQUIRE(after.nualt == before.nualt + 1);
   }

   SECTION("bypassed")
   {
      {
         InducedTrial trial(InducedTrial::History::BYPASS);
         nudge(19);
         energy(calc::v1);
         const Induced within;
         REQUIRE_FALSE(sameDipoles(within, before));
         REQUIRE(sameHistory(within, before));
      }
      const Induced after;
      REQUIRE(sameDipoles(after, before));
      REQUIRE(sameHistory(after, before));

      // and the solves after it add to the history again
      energy(calc::v1);
      REQUIRE(nualt == before.nualt + 1);
   }

   s.session.end();
}

namespace {
// Whether the system solves the second set of induced dipoles: AMOEBA does,
// and neither AMOEBA+ nor HIPPO does.
bool twoDipoleSets()
{
   return not polpot::use_tholed and not mplpot::use_chgpen;
}

// The largest induced dipole component on the first nlig atoms.
real largestDipole(int nlig)
{
   real big = 0;
   for (const real(*u)[3] : {tinker::uind, tinker::uinp}) {
      if (u == tinker::uinp and not twoDipoleSets())
         continue;
      const std::vector<real> v = grab3(u);
      for (int k = 0; k < 3 * nlig; ++k)
         big = std::max(big, std::abs(v[k]));
   }
   return big;
}

// Overwrites the first nlig atoms of every predictor slot with a dipole, as a
// history solved where those atoms were still polarizable would hold. The
// second set is picked by the model: an earlier system leaves its pointers
// dangling.
void plantLigandSlots(int nlig)
{
   for (int i = 0; i < maxualt; ++i) {
      for (real(**slot)[3] : {kUd[i], kUp[i]}) {
         if (slot == kUp[i] and not twoDipoleSets())
            continue;
         std::vector<real> v = grab3(*slot);
         for (int k = 0; k < 3 * nlig; ++k)
            v[k] = 0.1;
         darray::copyin(g::q0, n, *slot, reinterpret_cast<real(*)[3]>(v.data()));
      }
   }
   waitFor(g::q0);
}

// Zeroes the polarizability of the first nzero atoms, in the state each energy
// call rebuilds it from as well as in the state it leaves.
void zeroPolarity(int nzero)
{
   std::vector<real> pol(n), inv(n);
   darray::copyout(g::q0, n, pol.data(), polarity);
   darray::copyout(g::q0, n, inv.data(), polarity_inv);
   waitFor(g::q0);
   for (int i = 0; i < nzero; ++i) {
      pol[i] = 0;
      inv[i] = real(1) / real(1.0e-16);
   }
   darray::copyin(g::q0, n, polarity, pol.data());
   darray::copyin(g::q0, n, polarity_inv, inv.data());
   if (polarityorig)
      darray::copyin(g::q0, n, polarityorig, pol.data());
   waitFor(g::q0);
}

// An atom of zero polarizability keeps no induced dipole, whatever the
// predictor extrapolates for it: the solve never moves such a dipole, so the
// predicted guess must not leave one there. This holds in each solver, CUDA or
// OpenACC, whichever the build runs.
void zeroPolarityGuess(const std::string& keyextra, const System& sys)
{
   Setup s(kMd, keyextra, sys);
   const int nzero = 3;
   zeroPolarity(nzero);
   fillHistory();
   REQUIRE(largestDipole(nzero) == 0);
   REQUIRE(largestDipole(n) > 0);
   plantLigandSlots(nzero);
   REQUIRE(induceGuess() == UGuess::PREDICT);
   nudge(19);
   energy(calc::v1);
   REQUIRE(largestDipole(nzero) == 0);
   s.session.end();
}
}

TEST_CASE("INDUCE-zero-polarity-guess", "[ff][induce]")
{
   zeroPolarityGuess("", kWater);
}

TEST_CASE("INDUCE-zero-polarity-guess-hippo", "[ff][induce]")
{
   zeroPolarityGuess("polar-predict aspc\n", kHippo);
}

// Lambda dynamics needs the CUDA platform.
#if TINKER_GPULANG_CUDA
namespace {
// nextLmdaWin() reads and advances these.
struct WindowScope
{
   int win, nwin;
   std::vector<int> ends;
   std::vector<double> list;

   WindowScope()
      : win(lmdawin)
      , nwin(nlmdawin)
      , ends(lmdawinend)
      , list(lmdawinlist)
   {}

   ~WindowScope()
   {
      lmdawin = win;
      nlmdawin = nwin;
      lmdawinend = ends;
      lmdawinlist = list;
   }
};

// Single topology scales the ligand polarizabilities by the polarization lambda;
// with a lambda derivative to take, the ligand stays in the solve at zero, with
// no polarizability and so no induced dipoles. The history holds the dipoles
// of one polarization lambda: the first solve after lambda moves, whether by a
// window or any other way, starts over from the direct guess, and the predictor
// resumes once the history at the new lambda is full. The solve never moves an
// atom of zero polarizability, so a predicted guess must not leave a dipole
// there even if the history holds one.
void lambdaJump(const std::string& keyextra, const System& sys, int nlig)
{
   Setup s(kMd, "lambda 1.0\nlambda-deriv\n" + keyextra, sys);
   REQUIRE_FALSE(use_epdt);
   REQUIRE(use_dlmda);
   fillHistory();
   REQUIRE(plam == 1);
   REQUIRE(largestDipole(nlig) > 0);

   // Solves at the current lambda from an empty history: each one adds to it,
   // and the predictor takes over once it is full.
   auto refill = [nlig](int& step) {
      for (int k = 1; k < maxualt; ++k) {
         CAPTURE(k);
         REQUIRE(induceGuess() == UGuess::DIRECT);
         nudge(step++);
         energy(calc::v1);
         REQUIRE(nualt == k + 1);
         if (plam == 0)
            REQUIRE(largestDipole(nlig) == 0);
      }
      REQUIRE(induceGuess() == UGuess::PREDICT);
   };

   int step = 19;
   SECTION("lambda moves")
   {
      lambda = 0;
      nudge(step++);
      energy(calc::v1);
      REQUIRE(plam == 0);
      REQUIRE(nualt == 1);
      REQUIRE(largestDipole(nlig) == 0);
      refill(step);
      nudge(step++);
      energy(calc::v1);
      REQUIRE(largestDipole(nlig) == 0);

      // off zero the ligand is polarizable again, from the direct guess
      lambda = 0.01;
      nudge(step++);
      energy(calc::v1);
      REQUIRE(nualt == 1);
      REQUIRE(largestDipole(nlig) > 0);
      refill(step);
   }

   SECTION("next window")
   {
      WindowScope scope;
      nlmdawin = 2;
      lmdawinlist = {1.0, 0.0};
      lmdawinend = {20, 40};
      lmdawin = 1;
      nextLmdaWin();
      REQUIRE(lambda == 0);
      nudge(step++);
      energy(calc::v1);
      REQUIRE(nualt == 1);
      REQUIRE(largestDipole(nlig) == 0);
      refill(step);
   }

   SECTION("rejected trial at another lambda")
   {
      // the history in the first set of slots, which every model keeps
      auto history = []() {
         std::vector<std::vector<real>> h;
         for (int i = 0; i < maxualt; ++i)
            h.push_back(grab3(*kUd[i]));
         return h;
      };
      const int nu = nualt;
      const auto before = history();
      {
         InducedTrial trial(InducedTrial::History::USE);
         lambda = 0;
         nudge(step);
         energy(calc::v1);
         REQUIRE(nualt == 1);
      }
      lambda = 1;
      REQUIRE(nualt == nu);
      REQUIRE(history() == before);
      // and the solves at the lambda of the history carry on predicting
      REQUIRE(induceGuess() == UGuess::PREDICT);
      nudge(step++);
      energy(calc::v1);
      REQUIRE(nualt == (nu + 1 > 2 * maxualt ? nu + 1 - maxualt : nu + 1));
   }

   SECTION("stale ligand slots")
   {
      lambda = 0;
      nudge(step++);
      energy(calc::v1);
      refill(step);
      plantLigandSlots(nlig);
      nudge(step++);
      energy(calc::v1);
      REQUIRE(largestDipole(nlig) == 0);
   }

   s.session.end();
}
}

TEST_CASE("INDUCE-lambda-jump", "[ff][induce]")
{
   // the two water molecules of the ligand
   lambdaJump("", kWater, 6);
}

TEST_CASE("INDUCE-lambda-jump-hippo", "[ff][induce]")
{
   // the first dimethylsulfoxide molecule
   lambdaJump("polar-predict aspc\n", kHippo, 10);
}

// Dual topology keeps a predictor history for each endpoint. A trial that asks
// to use the history bypasses both instead, since putting a rejected trial back
// would take a slot from each; accepted or not, it leaves them as they were.
TEST_CASE("INDUCE-trial-dualtopo", "[ff][induce]")
{
   Setup s(kMd, "lambda-deriv2\n");
   REQUIRE(use_epdt);
   REQUIRE(polpred == UPred::ASPC);
   REQUIRE(induceHistoryUsable());
   for (int step = 0; step < 19; ++step) {
      nudge(step);
      energy(calc::v1);
   }
   DtPass pass[2];
   dtPasses(pass);
   std::vector<Induced> before;
   for (int k : {0, 1}) {
      REQUIRE(induceLoadHistory(pass[k].mask) >= maxualt);
      before.emplace_back();
      REQUIRE(before.back().nualt >= maxualt);
      REQUIRE(induceGuess() == UGuess::PREDICT);
   }
   REQUIRE_FALSE(sameHistory(before[0], before[1]));

   for (bool accept : {false, true}) {
      CAPTURE(accept);
      {
         InducedTrial trial(InducedTrial::History::USE);
         nudge(19);
         energy(calc::v1);
         if (accept)
            trial.accept();
      }
      for (int k : {0, 1}) {
         CAPTURE(k);
         induceLoadHistory(pass[k].mask);
         REQUIRE(sameHistory(Induced(), before[k]));
      }
   }

   s.session.end();
}

// One step at main lambda lmda with energy version vers, and the history
// lengths the next solves at the two dual topology endpoints would extrapolate
// from, with the guesses they would take.
struct EndpointStep
{
   int step = 0;
   int env = 0, coupled = 0;
   UGuess envGuess = UGuess::ZERO, guess = UGuess::ZERO;

   static UGuess guessFor(int len)
   {
      return len >= maxualt ? UGuess::PREDICT : UGuess::DIRECT;
   }

   void at(double lmda, int vers = calc::v1)
   {
      lambda = lmda;
      nudge(step++);
      energy(vers);
      DtPass pass[2];
      dtPasses(pass);
      env = induceLoadHistory(pass[0].mask);
      envGuess = guessFor(env);
      coupled = induceLoadHistory(pass[1].mask);
      guess = guessFor(coupled);
   }
};

// An endpoint history has to come from consecutive solves of one subsystem.
// The coupled endpoint carries nothing in the middle leg of a staged relative
// schedule, so a step there skips it and empties its history; the solves after
// the gap start from the direct guess and predict again only once the history
// is full. The legs on either side of the middle one charge different ligands,
// so a move that jumps over it, as a Monte Carlo lambda move may, empties the
// coupled history as well. The environment endpoint keeps predicting.
TEST_CASE("INDUCE-dualtopo-history-gaps", "[ff][induce]")
{
   const System rels = {"mutate/083_water_rels_ye_l085.key", "mutate/water2.xyz", "commit_6fe8e913/water03.prm"};
   Setup s(kMd, "polar-predict aspc\nrel-stage lig2 0.0 0.3\nrel-stage vdwm 0.3 0.7\n", rels);
   REQUIRE(use_epdt);
   REQUIRE(use_rel);
   REQUIRE(induceHistoryUsable());

   EndpointStep e;
   for (int i = 0; i < maxualt + 3; ++i)
      e.at(0.85);
   REQUIRE(relstage == RelStage::LIG1);
   REQUIRE(e.env >= maxualt);
   REQUIRE(e.coupled >= maxualt);
   REQUIRE(e.guess == UGuess::PREDICT);

   // the middle leg: no coupled solve, and its history is dropped
   int env0 = e.env;
   e.at(0.5);
   REQUIRE(relstage == RelStage::VDWM);
   REQUIRE(e.env == env0 + 1);
   REQUIRE(e.coupled == 0);

   // back in the ligand 1 leg it starts over from the direct guess, and
   // predicts once it is full
   for (int i = 0; i < maxualt - 1; ++i)
      e.at(0.85);
   REQUIRE(e.coupled == maxualt - 1);
   REQUIRE(e.guess == UGuess::DIRECT);
   e.at(0.85);
   REQUIRE(e.coupled == maxualt);
   REQUIRE(e.guess == UGuess::PREDICT);

   // straight into the ligand 2 leg: another ligand, another history
   env0 = e.env;
   e.at(0.15);
   REQUIRE(relstage == RelStage::LIG2);
   REQUIRE(e.env == env0 + 1);
   REQUIRE(e.coupled == 1);
   REQUIRE(e.guess == UGuess::DIRECT);

   s.session.end();
}

// Whether an endpoint is solved also depends on the version: at a weight of
// zero with a slope, the coupled endpoint is solved for the lambda derivative
// of a gradient and skipped for an energy alone, which empties its history.
TEST_CASE("INDUCE-dualtopo-version-skip", "[ff][induce]")
{
   Setup s(kMd, "lambda-deriv2\n");
   REQUIRE(use_epdt);
   REQUIRE(induceHistoryUsable());

   EndpointStep e;
   for (int i = 0; i < maxualt + 3; ++i)
      e.at(0.5);
   REQUIRE(e.coupled >= maxualt);

   int env0 = e.env, coupled0 = e.coupled;
   e.at(0.0);
   DtPass pass[2];
   dtPasses(pass);
   REQUIRE(pass[1].wa == 0);
   REQUIRE(pass[1].wb != 0);
   REQUIRE(e.env == env0 + 1);
   REQUIRE(e.coupled == coupled0 + 1);

   env0 = e.env;
   e.at(0.0, calc::v0);
   REQUIRE(e.env == env0 + 1);
   REQUIRE(e.coupled == 0);
   e.at(0.0);
   REQUIRE(e.coupled == 1);
   REQUIRE(e.guess == UGuess::DIRECT);

   s.session.end();
}

// An energy alone solves only the environment endpoint at lambda 0 and only the
// coupled one at lambda 1. The environment endpoint is the lambda 0 state, so
// while it predicts on its own the ligand keeps no dipoles. Each skipped
// endpoint starts over from the direct guess when both are solved again and
// predicts once its history is full.
TEST_CASE("INDUCE-dualtopo-endpoint-return", "[ff][induce]")
{
   Setup s(kMd, "lambda-deriv2\n");
   REQUIRE(use_epdt);
   REQUIRE(induceHistoryUsable());
   const int nlig = 6;

   EndpointStep e;
   for (int i = 0; i < maxualt + 3; ++i)
      e.at(0.5);
   REQUIRE(e.envGuess == UGuess::PREDICT);
   REQUIRE(e.guess == UGuess::PREDICT);

   // the environment endpoint alone; its pass leaves the dipoles
   const int env0 = e.env;
   for (int i = 1; i <= 3; ++i) {
      e.at(0.0, calc::v0);
      REQUIRE(e.coupled == 0);
      REQUIRE(e.env == env0 + i);
      REQUIRE(e.envGuess == UGuess::PREDICT);
      REQUIRE(largestDipole(nlig) == 0);
   }

   // the coupled endpoint alone
   e.at(1.0, calc::v0);
   REQUIRE(e.env == 0);
   REQUIRE(e.coupled == 1);
   REQUIRE(largestDipole(nlig) > 0);

   // both again: each predicts once its own history is full
   for (int i = 1; i < maxualt; ++i) {
      e.at(0.5);
      REQUIRE(e.env == i);
      REQUIRE(e.envGuess == UGuess::DIRECT);
   }
   REQUIRE(e.coupled == maxualt);
   REQUIRE(e.guess == UGuess::PREDICT);
   e.at(0.5);
   REQUIRE(e.env == maxualt);
   REQUIRE(e.envGuess == UGuess::PREDICT);

   s.session.end();
}

// A trial leaves no trace in either endpoint history, even when its energy
// alone skips one endpoint, as a Monte Carlo barostat trial at lambda 0 does,
// and even when the two histories are of different lengths.
TEST_CASE("INDUCE-dualtopo-trial-skip", "[ff][induce]")
{
   Setup s(kMd, "lambda-deriv2\n");
   REQUIRE(use_epdt);
   REQUIRE(induceHistoryUsable());

   EndpointStep e;
   for (int i = 0; i < maxualt + 3; ++i)
      e.at(0.5);
   // the environment history starts over at lambda 1, the coupled one keeps on
   e.at(1.0, calc::v0);
   e.at(0.5);
   e.at(0.5);
   const int env0 = e.env, coupled0 = e.coupled;
   REQUIRE(env0 == 2);
   REQUIRE(coupled0 >= maxualt);

   for (bool accept : {false, true}) {
      CAPTURE(accept);
      {
         InducedTrial trial(InducedTrial::History::USE);
         lambda = 0;
         nudge(e.step++);
         energy(calc::energy);
         if (accept)
            trial.accept();
      }
      DtPass pass[2];
      dtPasses(pass);
      CHECK(induceLoadHistory(pass[0].mask) == env0);
      CHECK(induceLoadHistory(pass[1].mask) == coupled0);
   }

   s.session.end();
}

// The endpoints of dual topology do not move with lambda, so both histories
// carry on as lambda leaves zero, and the predicted dipoles at the new lambda
// are those of a solve from the direct guess.
TEST_CASE("INDUCE-dualtopo-lambda-move", "[ff][induce]")
{
   Setup s(kMd, "lambda-deriv2\n");
   REQUIRE(use_epdt);
   REQUIRE(induceHistoryUsable());

   EndpointStep e;
   for (int i = 0; i < maxualt + 3; ++i)
      e.at(0.0);
   const int env0 = e.env, coupled0 = e.coupled;
   REQUIRE(e.envGuess == UGuess::PREDICT);
   REQUIRE(e.guess == UGuess::PREDICT);

   e.at(0.01);
   REQUIRE(e.env == env0 + 1);
   REQUIRE(e.coupled == coupled0 + 1);
   // the coupled endpoint is the last pass
   const std::vector<real> predicted = grab3(uind);
   std::vector<real> direct;
   {
      InducedTrial trial(InducedTrial::History::BYPASS);
      energy(calc::v1);
      direct = grab3(uind);
   }
   real big = 0, diff = 0;
   for (size_t k = 0; k < direct.size(); ++k) {
      big = std::max(big, std::abs(direct[k]));
      diff = std::max(diff, std::abs(predicted[k] - direct[k]));
   }
   REQUIRE(big > 0.01);
   REQUIRE(diff < 1.0e-5);

   s.session.end();
}
#endif

// HIPPO keeps one set of predictor slots, where AMOEBA keeps two; a trial must
// not reach for the second. Copying an unallocated slot is an error that only
// a build that checks the device calls reports, so this case counts in the
// debug build.
TEST_CASE("INDUCE-trial-hippo", "[ff][induce]")
{
   Setup s(kMd, "polar-predict aspc\n", kHippo);
   REQUIRE(mplpot::use_chgpen);
   REQUIRE_FALSE(polpot::use_tholed);
   fillHistory();

   struct State
   {
      std::vector<real> uind, udir;
      int nualt;
      std::vector<std::vector<real>> slots;
      State()
         : uind(grab3(tinker::uind))
         , udir(grab3(tinker::udir))
         , nualt(tinker::nualt)
      {
         for (int i = 0; i < maxualt; ++i)
            slots.push_back(grab3(*kUd[i]));
      }
   };
   auto same = [](const State& a, const State& b) {
      bool ok = sameBits(a.uind, b.uind) and sameBits(a.udir, b.udir) and a.nualt == b.nualt;
      for (size_t i = 0; i < a.slots.size(); ++i)
         ok = ok and sameBits(a.slots[i], b.slots[i]);
      return ok;
   };

   const State before;
   for (auto history : {InducedTrial::History::USE, InducedTrial::History::BYPASS}) {
      {
         InducedTrial trial(history);
         nudge(19);
         energy(calc::v1);
         REQUIRE_FALSE(sameBits(grab3(uind), before.uind));
      }
      REQUIRE(same(State(), before));
   }

   s.session.end();
}

// Exchange polarization scales the polarizabilities by matrices found from
// the structure, which the solve of a trial overwrites. They go back with the
// induced dipoles they belong to.
TEST_CASE("INDUCE-trial-expol", "[ff][induce]")
{
   Setup s(kMd, "polar-predict aspc\n", kExpol);
   REQUIRE(polpot::use_expol);
   fillHistory();

   // flat, as the snapshot copies them
   auto grab9 = [](real (*dev)[3][3]) {
      std::vector<real> v(9 * n);
      darray::copyout(g::q0, 9 * n, v.data(), &dev[0][0][0]);
      waitFor(g::q0);
      return v;
   };

   const auto scale0 = grab9(polscale), inv0 = grab9(polinv);
   for (auto history : {InducedTrial::History::USE, InducedTrial::History::BYPASS}) {
      {
         InducedTrial trial(history);
         nudge(19);
         energy(calc::v1);
         REQUIRE_FALSE(sameBits(grab9(polscale), scale0));
         REQUIRE_FALSE(sameBits(grab9(polinv), inv0));
      }
      REQUIRE(sameBits(grab9(polscale), scale0));
      REQUIRE(sameBits(grab9(polinv), inv0));
   }

   s.session.end();
}

// A solver that throws inside a trial must not leave the bypass switched on
// for whatever the process runs next.
TEST_CASE("INDUCE-trial-unwinds", "[ff][induce]")
{
   Setup s(kMd);
   fillHistory();
   const Induced before;

   auto failingTrial = []() {
      InducedTrial trial(InducedTrial::History::BYPASS);
      nudge(19);
      energy(calc::v1);
      induceNotConverged(udir, udirp);
   };
   REQUIRE_THROWS_AS(failingTrial(), FatalError);

   REQUIRE_FALSE(induceHistoryBypassed());
   REQUIRE(induceGuess() == UGuess::PREDICT);
   const Induced after;
   REQUIRE(sameDipoles(after, before));
   REQUIRE(sameHistory(after, before));

   // the next trial starts as the first did
   {
      InducedTrial trial(InducedTrial::History::USE);
      REQUIRE(induceGuess() == UGuess::PREDICT);
   }

   s.session.end();
}

TEST_CASE("INDUCE-trial-no-polar", "[ff][induce]")
{
   Setup s(kMd, "polarizeterm none\n");
   REQUIRE_FALSE(use(Potent::POLAR));
   const double eps = testGetEps(1.0e-4, 1.0e-8);

   energy(calc::v1);
   const double e0 = esum;
   {
      InducedTrial trial(InducedTrial::History::USE);
      energy(calc::v1);
      COMPARE_REALS(esum, e0, eps);
   }
   {
      InducedTrial trial(InducedTrial::History::BYPASS);
      REQUIRE(induceHistoryBypassed());
      energy(calc::v1);
      COMPARE_REALS(esum, e0, eps);
   }
   REQUIRE_FALSE(induceHistoryBypassed());

   s.session.end();
}

// The snapshot a trial keeps: the slot of the history a solve would overwrite
// is part of it only when the solves to come can write the history.
TEST_CASE("INDUCE-snapshot-slot", "[ff][induce]")
{
   Setup s(kMd);
   fillHistory();
   const Induced before;
   const int pos = nualt % maxualt;

   for (bool history : {true, false}) {
      CAPTURE(history);
      InducedSnapshot snapshot;
      snapshot.save(history);

      // what a solve at a trial geometry would leave behind
      darray::zero(g::q0, n, uind, uinp, udir, udirp, *kUd[pos], *kUp[pos]);
      waitFor(g::q0);
      nualt += 1;

      snapshot.restore();
      const Induced after;
      REQUIRE(sameDipoles(after, before));
      REQUIRE(after.nualt == before.nualt);
      for (int i = 0; i < maxualt; ++i) {
         if (i == pos and not history)
            continue;
         REQUIRE(sameBits(after.slots[i], before.slots[i]));
         REQUIRE(sameBits(after.pslots[i], before.pslots[i]));
      }
      if (not history) {
         const std::vector<real> zeros(3 * n, 0);
         REQUIRE(sameBits(after.slots[pos], zeros));
         REQUIRE(sameBits(after.pslots[pos], zeros));
      }
   }

   s.session.end();
}
