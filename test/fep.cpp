#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/efep.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/edisp.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/darray.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"

#include "test.h"
#include "testrt.h"

#include <tinker/detail/bath.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/dsppot.hh>
#include <tinker/detail/fep.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/uprior.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Tests of free energy perturbation over lambda windows (src/efep.cpp),
// following the Fortran suite in tinker/test/test_fep.f. The method relies on
// one thing above all: an energy found at a trial lambda from a running system
// is the energy of that structure set up afresh at that lambda. The trial
// cases check exactly that for each kind of term, and the state cases check
// that finding it leaves the running system alone.
//
// Lambda dynamics needs the CUDA platform (dlmdaData2).

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
// A system and the key file it is set up from, under test/file.
struct Fixture
{
   const char* tag;
   const char* key;    // key file
   const char* xyz;    // coordinates
   const char* prm;    // parameter file
   const char* native; // key lines the native setup adds
   const char* fresh;  // key lines every fresh setup adds ahead of its lambda
   std::vector<double> ltrial;
   double eps;      // mixed precision tolerance on a trial energy, kcal/mol
   void (*terms)(); // checks that the terms the case is about are in use
};

// Free energy perturbation from the lambda the key file names, 0.5; a second
// window makes the schedule a legal one.
const char* kFepHalf = "\nlambda-mode fep\nlambda-window 0.5\nlambda-window 0.4\n";

struct Setup
{
   TestLmdaFlagReset lmdaReset;
   std::string xyz, key;
   const char* argv[4];
   TestFile fxyz, fkey, fprm;
   TestSession session;

   Setup(const Fixture& fx, int rc, const std::string& keyextra, bool init = true)
      : xyz(testBaseName(fx.xyz))
      , key(testBaseName(fx.key))
      , argv{"dummy", xyz.c_str(), "-k", key.c_str()}
      , fxyz(std::string(TINKER9_DIRSTR "/test/file/") + fx.xyz, xyz)
      , fkey(std::string(TINKER9_DIRSTR "/test/file/") + fx.key, key, keyextra)
      , fprm(std::string(TINKER9_DIRSTR "/test/file/") + fx.prm)
      , session(4, argv)
   {
      rc_flag = rc;
      if (init)
         session.init();
   }
};

// With the analysis buffers, each term keeps its own energy and the separate
// multipole and polarization kernels run. Dynamics runs without them, which on
// this platform puts plain AMOEBA on the fused multipole and polarization
// kernel, and partial charges with Lennard-Jones on theirs. A trial energy has
// to be right on both routes, so the trial cases take each in turn; the fresh
// setups they are compared with always take the first.
const int kRc = calc::xyz | calc::mass | calc::vmask;
const int kRcDyn = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;

std::string lambdaKey(double lmda)
{
   char buf[64];
   std::snprintf(buf, sizeof(buf), "\nlambda %.8f\n", lmda);
   return buf;
}

std::vector<real> grab3(const real (*dev)[3])
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real (*)[3]>(v.data()), dev);
   waitFor(g::q0);
   return v;
}

template <class T>
std::vector<T> grab1(const T* dev)
{
   std::vector<T> v(n);
   darray::copyout(g::q0, n, v.data(), dev);
   waitFor(g::q0);
   return v;
}

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b)
{
   return a.size() == b.size() and 0 == std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}

// The trial energies of a fixture, found twice from its native setup with
// \c rc as its flags, against the energy of a fresh setup at each trial lambda.
void runTrial(const Fixture& fx, int rc = kRc)
{
   const double eps = testGetEps(fx.eps, 1.0e-6);
   const int nt = (int)fx.ltrial.size();
   std::vector<double> e1(nt), e2(nt);
   double en1 = 0, en2 = 0, lnative = 0;

   {
      Setup s(fx, rc, fx.native);
      lnative = lambda;
      if (fx.terms)
         fx.terms();

      // get the trial energies twice from the native setup
      efepTrial(nt, fx.ltrial.data(), e1.data(), en1);
      REQUIRE(lambda == lnative);
      REQUIRE(esum == en1);
      efepTrial(nt, fx.ltrial.data(), e2.data(), en2);
      REQUIRE(lambda == lnative);
      REQUIRE(esum == en2);
      COMPARE_REALS(en2, en1, eps);
      for (int i = 0; i < nt; ++i) {
         CAPTURE(fx.ltrial[i]);
         COMPARE_REALS(e2[i], e1[i], eps);
      }

      // the native energy is that of an ordinary energy call
      energy(calc::energy);
      COMPARE_REALS(esum, en2, eps);
      s.session.end();
   }

   // set the system up again at each trial lambda for a reference
   for (int i = 0; i < nt; ++i) {
      CAPTURE(fx.ltrial[i]);
      Setup s(fx, kRc, fx.fresh + lambdaKey(fx.ltrial[i]));
      REQUIRE_FALSE(use_fep);
      COMPARE_REALS(lambda, fx.ltrial[i], 1.0e-8);
      energy(calc::energy);
      COMPARE_REALS(e1[i], esum, eps);
      COMPARE_REALS(e2[i], esum, eps);
      s.session.end();
   }
}

// The energy of a system on its own, set up from a key file and coordinates
// under test/file with \c keyextra added.
double energyOf(const char* key, const char* xyz, const char* prm, const std::string& keyextra = "")
{
   const Fixture f = {"", key, xyz, prm, "", "", {}, 0, nullptr};
   Setup s(f, kRc, keyextra);
   energy(calc::energy);
   double e = esum;
   s.session.end();
   return e;
}

// The trial energy of a fixture at lambda 0 against the same physics without the
// mutation: the systems \c refs name, each set up on its own, sum to it
// (test_fep.f: fepend).
void runEnd(const Fixture& fx, std::vector<std::pair<const char*, const char*>> refs, double eps)
{
   double l0 = 0, e0 = 0, en = 0;
   {
      Setup s(fx, kRc, fx.native);
      efepTrial(1, &l0, &e0, en);
      s.session.end();
   }
   double eref = 0;
   for (const auto& r : refs)
      eref += energyOf(r.first, r.second, fx.prm);
   COMPARE_REALS(e0, eref, eps);
}

// With the soft core switch turned off, every dispersion pair and reciprocal
// share of the ligand scales with the one lambda power v, so the energy is
// linear in v (test_fep.f: displinear).
void runLinear(const char* key, double vl, double eps)
{
   auto e = [key](double l) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "\nvdw-lambda %.8f\n", l);
      return energyOf(key, "dmso/dmso.xyz", "hippo/hippo19.prm", buf);
   };
   const double e0 = e(0), e1 = e(1), ev = e(vl);
   const double v = vl * vl * vl * vl;
   COMPARE_REALS(ev, e0 + v * (e1 - e0), eps);
}

// Free energy perturbation moves the main lambda with no lambda derivative.
void termsFep()
{
   REQUIRE(use_fep);
   REQUIRE(use_mainlmda);
   REQUIRE_FALSE(use_dlmda);
}

void termsWindow()
{
   termsFep();
   REQUIRE(use(Potent::MPOLE));
   REQUIRE(use(Potent::POLAR));
}

void termsMpole()
{
   termsFep();
   REQUIRE(use(Potent::MPOLE));
   REQUIRE_FALSE(use(Potent::POLAR));
}

void termsCharge()
{
   termsFep();
   REQUIRE(use(Potent::CHARGE));
}

void termsHippo()
{
   termsFep();
   REQUIRE(mplpot::use_chgpen);
   REQUIRE(use(Potent::POLAR));
   REQUIRE(use(Potent::REPULS));
   REQUIRE(use(Potent::DISP));
   REQUIRE(use(Potent::CHGTRN));
}

void termsAplus()
{
   termsFep();
   REQUIRE(mplpot::use_chgpen);
   REQUIRE(polpot::use_tholed);
   REQUIRE(use(Potent::POLAR));
   REQUIRE(use(Potent::CHGFLX));
}

// A single AMOEBA+ water hydrogen whose flux bond reaches into the environment:
// the flux moves the net charge, and the background term with it.
void termsAplusCross()
{
   termsFep();
   REQUIRE(mplpot::use_chgpen);
   REQUIRE(use(Potent::CHGFLX));
   REQUIRE_FALSE(use(Potent::POLAR));
   REQUIRE(cfluxMovesNetCharge());
}

// HIPPO polarization with exchange polarization, whose sizes follow the
// polarizability.
void termsExpol()
{
   REQUIRE(polpot::use_expol);
   REQUIRE(use(Potent::POLAR));
   REQUIRE_FALSE(use_epdt);
}

void termsExpolDt()
{
   REQUIRE(polpot::use_expol);
   REQUIRE(use_epdt);
}

// Ewald dispersion alone, whose reciprocal and self terms follow the van der
// Waals lambda.
void termsDisp()
{
   REQUIRE(useDEwald());
   REQUIRE_FALSE(use(Potent::MPOLE));
   REQUIRE_FALSE(use(Potent::VDW));
}

// Dispersion without Ewald, whose long range correction follows the lambda
// power of the soft core.
void termsDispCorr()
{
   REQUIRE_FALSE(useDEwald());
   REQUIRE(dsppot::use_dcorr);
   REQUIRE_FALSE(use(Potent::VDW));
}

#define WATER "mutate/water2.xyz", "commit_6fe8e913/water03.prm"
const std::vector<double> kAbs{0.0, 0.4, 0.6, 1.0};
const std::vector<double> kDual{0.0, 0.5, 0.7, 1.0};
const std::vector<double> kStaged{0.0, 0.15, 0.5, 0.7, 0.8, 0.9, 1.0};

const Fixture kFixtures[] = {
   // absolute single topology with all three sublambdas mapped
   {"ast", "mutate/164_water_ast_ne_mcut_d1_l05.key", WATER, "", "", kAbs, 1.0e-5, nullptr},
   {"ast-ewald-ion", "mutate/179_ionwat_ast_l05.key", "mutate/ionwat.xyz", "commit_ebe3611e/amoeba09.prm", "", "",
      kAbs, 1.0e-5, nullptr},
   // van der Waals annihilation with its long range correction
   {"ast-vcorr", "mutate/170_water_ast_vcorr_annih_d1_l05.key", WATER, "", "", kAbs, 1.0e-5, nullptr},
   // the quintic taper map of the electrostatic sublambda
   {"ast-qnt", "mutate/093_water_lmda_qnt_l05.key", WATER, "", "", kAbs, 1.0e-5, nullptr},
   // absolute dual topology polarization, with and without Ewald
   {"adt-ewald", "mutate/169_water_adt_d1_x2_l06.key", WATER, "", "", kDual, 1.0e-5, nullptr},
   {"adt", "mutate/178_water_adt_d1_ne_l06.key", WATER, "", "", kDual, 1.0e-5, nullptr},
   // staged relative, native in the last leg with trials in every leg, on a
   // stage boundary and at both ends
   {"rels-dt", "mutate/190_water_rels3_l085.key", WATER, "", "", kStaged, 1.0e-5, nullptr},
   {"rels-st", "mutate/196_water_rels3_st_l085.key", WATER, "", "", kStaged, 1.0e-5, nullptr},
   // free energy perturbation itself, which moves the main lambda with no
   // lambda derivative in use: mutual polarization, and multipoles alone
   {"window", "fep/window.key", WATER, "", "\nlambda-mode none\n", kAbs, 1.0e-5, termsWindow},
   {"window-mpole", "fep/window.key", WATER, "\npolarizeterm none\n", "\nlambda-mode none\npolarizeterm none\n",
      kAbs, 1.0e-5, termsMpole},
   // The kinds of term the water cases do not use: partial charges, the HIPPO
   // terms, and AMOEBA+ with charge flux. Their parameters are scaled in single
   // precision here and in double by the Fortran setup, which shows in the
   // larger energies away from the ends of the lambda range.
   {"charge", "fep/charge.key", "trpcage/trp_charmm.xyz", "commit_11e84c69/charmm19.prm", kFepHalf, "", kAbs,
      5.0e-5, termsCharge},
   {"hippo", "fep/hippo.key", "dmso/dmso.xyz", "hippo/hippo19.prm", kFepHalf, "", kAbs, 2.0e-4, termsHippo},
   {"aplus", "fep/aplus.key", "aplus2022/tetramer.xyz", "aplus2022/AMOEBAplus_Org.prm", kFepHalf, "", kAbs,
      1.0e-5, termsAplus},
   {"aplus-cross", "mutate/209_apgas_cross_ewald_nold_l04.key", "mutate/apgas.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm", kFepHalf, "", kAbs, 1.0e-5, termsAplusCross},
   // exchange polarization of an Na+ water cluster, the ion mutated in single
   // and dual topology, and of a box with ions, a water mutated under Ewald
   {"expol", "fep/expol.key", "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm", "", "", kAbs, 1.0e-4,
      termsExpol},
   {"expol-dt", "fep/expoldt.key", "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm", "", "", kAbs, 1.0e-4,
      termsExpolDt},
   {"expol-box", "fep/expolbox.key", "testlmda/expbox.xyz", "commit_dd1761ac/expol.prm", "", "", kAbs, 1.0e-4,
      termsExpol},
   // Ewald dispersion of one DMSO of a DMSO box, annihilated or decoupled
   {"dispann", "fep/dispann.key", "dmso/dmso.xyz", "hippo/hippo19.prm", "", "", kAbs, 1.0e-4, termsDisp},
   {"dispdec", "fep/dispdec.key", "dmso/dmso.xyz", "hippo/hippo19.prm", "", "", kAbs, 1.0e-4, termsDisp},
   // dispersion with its long range correction (test_testlmda.f case 113)
   {"dispcorr", "testlmda/113_hipmix_dsp_corr_l04.key", "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm", "", "",
      kAbs, 1.0e-4, termsDispCorr},
};

const Fixture& fx(const char* tag)
{
   for (const auto& f : kFixtures)
      if (std::string(f.tag) == tag)
         return f;
   FAIL("no fep fixture named " << tag);
   return kFixtures[0];
}

// Everything a trial energy must leave alone.
struct State
{
   double lambda, elam, plam, vlam;
   Box box;
   std::vector<pos_prec> x, y, z;
   std::vector<grad_prec> gx, gy, gz;
   std::vector<real> uind, uinp, udir, udirp, prepep;
   int nualt;
   std::vector<std::vector<real>> slots, pslots;

   State()
      : lambda(tinker::lambda)
      , elam(tinker::elam)
      , plam(tinker::plam)
      , vlam(tinker::vlam)
      , x(grab1(xpos))
      , y(grab1(ypos))
      , z(grab1(zpos))
      , gx(grab1(tinker::gx))
      , gy(grab1(tinker::gy))
      , gz(grab1(tinker::gz))
      , nualt(tinker::nualt)
   {
      boxGetCurrent(box);
      // the exchange polarization sizes follow the polarizability
      if (polpot::use_expol)
         prepep = grab1(tinker::prepep);
      if (use(Potent::POLAR)) {
         // HIPPO keeps the one set of induced dipoles
         const bool hippo = mplpot::use_chgpen and not polpot::use_tholed;
         uind = grab3(tinker::uind);
         udir = grab3(tinker::udir);
         if (not hippo) {
            uinp = grab3(tinker::uinp);
            udirp = grab3(tinker::udirp);
         }
         real(*ds[])[3] = {udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, udalt_08,
            udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15};
         real(*ps[])[3] = {upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07, upalt_08,
            upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15};
         for (int i = 0; i < maxualt; ++i) {
            slots.push_back(grab3(ds[i]));
            if (not hippo)
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
   const real3 va[] = {a.box.lvec1, a.box.lvec2, a.box.lvec3, a.box.recipa, a.box.recipb, a.box.recipc};
   const real3 vb[] = {b.box.lvec1, b.box.lvec2, b.box.lvec3, b.box.recipa, b.box.recipb, b.box.recipc};
   for (int i = 0; i < 6; ++i) {
      REQUIRE(va[i].x == vb[i].x);
      REQUIRE(va[i].y == vb[i].y);
      REQUIRE(va[i].z == vb[i].z);
   }
   REQUIRE(sameBits(a.x, b.x));
   REQUIRE(sameBits(a.y, b.y));
   REQUIRE(sameBits(a.z, b.z));
   REQUIRE(sameBits(a.gx, b.gx));
   REQUIRE(sameBits(a.gy, b.gy));
   REQUIRE(sameBits(a.gz, b.gz));
   REQUIRE(sameBits(a.uind, b.uind));
   REQUIRE(sameBits(a.uinp, b.uinp));
   REQUIRE(sameBits(a.udir, b.udir));
   REQUIRE(sameBits(a.udirp, b.udirp));
   REQUIRE(sameBits(a.prepep, b.prepep));
   REQUIRE(a.nualt == b.nualt);
   REQUIRE(a.slots.size() == b.slots.size());
   for (size_t i = 0; i < a.slots.size(); ++i)
      REQUIRE(sameBits(a.slots[i], b.slots[i]));
   REQUIRE(a.pslots.size() == b.pslots.size());
   for (size_t i = 0; i < a.pslots.size(); ++i)
      REQUIRE(sameBits(a.pslots[i], b.pslots[i]));
}

std::vector<double> gradient()
{
   std::vector<double> gxyz(3 * n);
   copyGradient(calc::grad, gxyz.data(), gxyz.data() + n, gxyz.data() + 2 * n);
   return gxyz;
}

double maxDiff(const std::vector<double>& a, const std::vector<double>& b)
{
   double d = 0;
   for (size_t i = 0; i < a.size(); ++i)
      d = std::max(d, std::fabs(a[i] - b[i]));
   return d;
}

// The lambdas, the structure, the gradient and the induced dipoles of a fixture
// across a call that finds its trial energies, and the energy and gradient of
// the dynamics step that comes next, with the flags a dynamics run has.
void runState(const Fixture& fx)
{
   const double eps = testGetEps(fx.eps, 1.0e-6);
   // the energy-only kernels against the ones that also find the gradient
   const double eps_step = testGetEps(std::max(fx.eps, 5.0e-4), 1.0e-6);
   const double eps_g = testGetEps(1.0e-4, 1.0e-6);
   const int nt = (int)fx.ltrial.size();
   std::vector<double> e(nt);
   double en = 0;

   // the energy and gradient call of a dynamics step
   Setup s(fx, kRcDyn, fx.native);
   energy(calc::v4);
   const double estep = esum;
   const std::vector<double> gstep = gradient();
   const State before;

   // The native energy is left in place of that of the step, found by the
   // energy-only call the trial values use.
   efepTrial(nt, fx.ltrial.data(), e.data(), en);
   requireSame(State(), before);
   REQUIRE(esum == en);
   COMPARE_REALS(en, estep, eps_step);

   // The parameter arrays were left to be rescaled; the next step reads them
   // at the native lambda all the same.
   energy(calc::v4);
   COMPARE_REALS(esum, estep, eps);
   REQUIRE(maxDiff(gradient(), gstep) < eps_g);

   s.session.end();
}

// Whether a line of a trial energy file is the one that labels its columns.
bool fepColumns(const std::string& line)
{
   return line.find("Trial Lambda") != std::string::npos and line.find("Step") != std::string::npos;
}

// Whether a line of the file contains text.
bool fepHasLine(const std::string& fname, const std::string& text)
{
   std::ifstream fs(fname);
   std::string line;
   while (std::getline(fs, line))
      if (line.find(text) != std::string::npos)
         return true;
   return false;
}

// Counts the energy records of a trial energy file: the nonblank lines after
// the one that labels the columns.
int fepCountRows(const std::string& fname)
{
   std::ifstream fs(fname);
   std::string line;
   bool header = false;
   int nrow = 0;
   while (std::getline(fs, line)) {
      if (not header)
         header = fepColumns(line);
      else if (line.find_first_not_of(" \t\r") != std::string::npos)
         ++nrow;
   }
   return nrow;
}

// The nth (1-based) energy record, counted the same way, and its box volume
// if vol is not null.
bool fepRow(const std::string& fname, int nth, int& istep, double& simlmda, double& trial, double& ene,
   double* vol = nullptr)
{
   std::ifstream fs(fname);
   std::string line;
   bool header = false;
   int nrow = 0;
   while (std::getline(fs, line)) {
      if (not header) {
         header = fepColumns(line);
         continue;
      }
      if (line.find_first_not_of(" \t\r") == std::string::npos)
         continue;
      if (++nrow == nth) {
         std::istringstream ss(line);
         ss >> istep >> simlmda >> trial >> ene;
         if (vol)
            return bool(ss >> *vol);
         return true;
      }
   }
   return false;
}

const int kMd = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

// Puts the base name back and removes the trial energy files of a case as it
// leaves scope, so that a failed check leaves neither to the cases after it.
struct FepFiles
{
   char savedname[240];
   int savedleng;
   std::vector<std::string> made;

   FepFiles(const char* base)
   {
      std::memcpy(savedname, files::filename, 240);
      savedleng = files::leng;
      FstrView fname = files::filename;
      fname = base;
      files::leng = (int)std::strlen(base);
   }

   // prtfephead claims a new version, so take the name it actually used
   std::string claim()
   {
      made.push_back(FstrView(dlmda::lmdasavefile).trim());
      return made.back();
   }

   ~FepFiles()
   {
      try {
         for (const auto& f : made)
            if (f.size())
               fileExistsAndDelete(f);
         std::memcpy(files::filename, savedname, 240);
         files::leng = savedleng;
      } catch (...) {
      }
   }
};

// How a run of dynamics over the windows of a fixture is set up.
struct DynOpts
{
   const char* fixture = "window";
   std::string extra;                     // key lines the case adds to the native ones
   BarostatEnum baro = BarostatEnum::NONE;
   bool sample = true;                    // find the trial energies
   bool check = false;                    // hold each native energy to an energy call
   bool propagate = false;                // leave the steps to mdPropagate()
   int iwrite = 1000;                     // steps between saved frames
   UPred pred = UPred::ASPC;              // the predictor the dynamics keeps
   void (*terms)() = nullptr;             // checks that the setup is the one the case is about
};

// Twenty four steps of dynamics over the three windows of the fixture, and
// the coordinates, velocities and trial energy records they end at.
struct DynEnd
{
   std::vector<pos_prec> x0, x, y, z;
   std::vector<vel_prec> vx, vy, vz;
   Box box0, box;
   int nfep;
   std::vector<int> fepstep;
   std::vector<double> feplmda, feptrial, fepene, fepvol;
   std::vector<RelStage> legs; // the active leg as each sample was taken
   int nmoved = 0;             // steps that changed the box
   int nmovedsample = 0;       // those sampled with steps of the run left to take
   double lmdastart = 0;       // the Fortran main lambda as the steps begin
   double lmdasaved = 0;       // and as the run ends, set by the frames saved
   double lmdalast = 0;        // the lambda of the last window
};

bool sameBox(const Box& a, const Box& b)
{
   const real3 va[] = {a.lvec1, a.lvec2, a.lvec3}, vb[] = {b.lvec1, b.lvec2, b.lvec3};
   for (int i = 0; i < 3; ++i)
      if (va[i].x != vb[i].x or va[i].y != vb[i].y or va[i].z != vb[i].z)
         return false;
   return true;
}

// Whether two runs ended at the same coordinates and velocities, bit for bit.
bool sameEnd(const DynEnd& a, const DynEnd& b)
{
   return sameBits(a.x, b.x) and sameBits(a.y, b.y) and sameBits(a.z, b.z) and sameBits(a.vx, b.vx)
      and sameBits(a.vy, b.vy) and sameBits(a.vz, b.vz);
}

// Samples each window twice, in the middle and on its last step. A step taken
// in a window after a sample of that window is what shows a trial lambda, or
// the parameters scaled to it, left behind; after the last step the next
// window sets its own.
const char* kTwice = "\nlambda-interval 2\n";

// With \c check, each native energy recorded is compared with an energy call
// made for the structure as the step left it. With \c propagate the steps are
// those of mdPropagate() and the integrator the session set up, as in a
// dynamics run, and not of a loop written out here.
DynEnd runDyn(const DynOpts& o)
{
   const Fixture& f = fx(o.fixture);
   const int nstep = 24;
   const double dt_ps = 0.0002;
   DynEnd end;

   Setup s(f, kMd, std::string(f.native) + o.extra + "\nintegrator verlet\nrandomseed 314159\n", false);
   // The session sets up an integrator of its own, which mdPropagate() drives;
   // it is to have no thermostat there, as the one made below has none.
   if (o.propagate)
      testMdInit(0., 0.);
   else
      testMdInit(298., o.baro == BarostatEnum::NONE ? 0. : 1.);
   s.session.init();
   if (o.terms)
      o.terms();
   // a stiff coupling, so that each step rescales the box by a visible amount
   const double oldtaupres = bath::taupres;
   if (o.baro != BarostatEnum::NONE)
      bath::taupres = 0.001;
   FepFiles files("fepdyn_tmp");
   // mdPropagate() saves the restart file of the run as it returns
   TestRemoveFileOnExit fdyn("fepdyn_tmp.dyn");
   // and its saved frames go to the archive
   TestRemoveFileOnExit farc("fepdyn_tmp.arc");
   int oldiwrite = inform::iwrite;
   inform::iwrite = o.iwrite;

   // the dynamics keeps the predictor the key file asks for
   REQUIRE(polpred == o.pred);

   // Start from rest: the velocities the setup draws differ from one session
   // of a process to the next.
   darray::zero(g::q0, n, tinker::vx, tinker::vy, tinker::vz);
   waitFor(g::q0);

   init_fepdyn(nstep);
   files.claim();
   REQUIRE(nfeptot > 0);
   end.x0 = grab1(xpos);
   boxGetCurrent(end.box0);
   end.lmdastart = mutant::lambda;
   end.lmdalast = lmdawinlist[nlmdawin - 1];
   if (o.propagate) {
      REQUIRE(o.sample);
      REQUIRE(o.baro == BarostatEnum::NONE);
      REQUIRE_FALSE(bath::isothermal);
      // The integrator made below starts with a force call of its own, which
      // adds a solve to the predictor history; make the same call here.
      VerletIntegrator::KickOff();
      mdPropagate(nstep, dt_ps);
      end.lmdasaved = mutant::lambda;
   } else {
      std::unique_ptr<BasicIntegrator> intg(new VerletIntegrator(ThermostatEnum::NONE, o.baro));
      Box before = end.box0;
      for (int istep = 1; istep <= nstep; ++istep) {
         intg->dynamic(istep, dt_ps);
         Box after;
         boxGetCurrent(after);
         const bool moved = not sameBox(after, before);
         before = after;
         const int nfep0 = nfep;
         if (o.sample)
            lmdaWinDyn(istep);
         if (nfep > nfep0)
            end.legs.push_back(relstage);
         if (moved) {
            end.nmoved += 1;
            if (efepSampleStep(istep) and istep < nstep)
               end.nmovedsample += 1;
         }
         if (o.check and nfep > nfep0) {
            // as a trial, so that the solve leaves no trace in the dynamics
            InducedTrial trial(InducedTrial::History::BYPASS);
            nblistRefresh();
            energy(calc::energy);
            COMPARE_REALS(fepene[nfep0], esum, testGetEps(f.eps, 1.0e-6));
            // every record of the sample has the volume of the box the step
            // ended at
            for (int i = nfep0; i < nfep; ++i)
               REQUIRE(fepvol[i] == boxVolume());
         }
         lmdaWinStep(istep);
      }
   }
   end.nfep = nfep;
   end.fepstep.assign(fepstep.begin(), fepstep.begin() + nfep);
   end.feplmda.assign(feplmda.begin(), feplmda.begin() + nfep);
   end.feptrial.assign(feptrial.begin(), feptrial.begin() + nfep);
   end.fepene.assign(fepene.begin(), fepene.begin() + nfep);
   end.fepvol.assign(fepvol.begin(), fepvol.begin() + nfep);
   boxGetCurrent(end.box);
   end.x = grab1(xpos);
   end.y = grab1(ypos);
   end.z = grab1(zpos);
   end.vx = grab1(tinker::vx);
   end.vy = grab1(tinker::vy);
   end.vz = grab1(tinker::vz);

   inform::iwrite = oldiwrite;
   bath::taupres = oldtaupres;
   s.session.end();
   return end;
}
}

TEST_CASE("FEP-trial-ast", "[ff][fep]") { runTrial(fx("ast")); }
TEST_CASE("FEP-trial-ast-ewald-ion", "[ff][fep]") { runTrial(fx("ast-ewald-ion")); }
TEST_CASE("FEP-trial-ast-vcorr", "[ff][fep]") { runTrial(fx("ast-vcorr")); }
TEST_CASE("FEP-trial-ast-qnt", "[ff][fep]") { runTrial(fx("ast-qnt")); }
TEST_CASE("FEP-trial-adt-ewald", "[ff][fep]") { runTrial(fx("adt-ewald")); }
TEST_CASE("FEP-trial-adt", "[ff][fep]") { runTrial(fx("adt")); }
TEST_CASE("FEP-trial-rels-dt", "[ff][fep]") { runTrial(fx("rels-dt")); }
TEST_CASE("FEP-trial-rels-st", "[ff][fep]") { runTrial(fx("rels-st")); }
TEST_CASE("FEP-trial-window", "[ff][fep]") { runTrial(fx("window")); }
TEST_CASE("FEP-trial-window-mpole", "[ff][fep]") { runTrial(fx("window-mpole")); }
TEST_CASE("FEP-trial-charge", "[ff][fep]") { runTrial(fx("charge")); }
TEST_CASE("FEP-trial-hippo", "[ff][fep]") { runTrial(fx("hippo")); }
TEST_CASE("FEP-trial-aplus", "[ff][fep]") { runTrial(fx("aplus")); }
TEST_CASE("FEP-trial-aplus-cross", "[ff][fep][cflux]") { runTrial(fx("aplus-cross")); }
TEST_CASE("FEP-trial-dispann", "[ff][fep][disp]") { runTrial(fx("dispann")); }
TEST_CASE("FEP-trial-expol", "[ff][fep][expol]") { runTrial(fx("expol")); }
TEST_CASE("FEP-trial-expol-dt", "[ff][fep][expol]") { runTrial(fx("expol-dt")); }
TEST_CASE("FEP-trial-expol-box", "[ff][fep][expol]") { runTrial(fx("expol-box")); }
TEST_CASE("FEP-trial-dispdec", "[ff][fep][disp]") { runTrial(fx("dispdec")); }
TEST_CASE("FEP-trial-dispcorr", "[ff][fep][disp]") { runTrial(fx("dispcorr")); }

TEST_CASE("FEP-trial-dyn-ast", "[ff][fep]") { runTrial(fx("ast"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-ast-ewald-ion", "[ff][fep]") { runTrial(fx("ast-ewald-ion"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-ast-vcorr", "[ff][fep]") { runTrial(fx("ast-vcorr"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-ast-qnt", "[ff][fep]") { runTrial(fx("ast-qnt"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-adt-ewald", "[ff][fep]") { runTrial(fx("adt-ewald"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-adt", "[ff][fep]") { runTrial(fx("adt"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-rels-dt", "[ff][fep]") { runTrial(fx("rels-dt"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-rels-st", "[ff][fep]") { runTrial(fx("rels-st"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-window", "[ff][fep]") { runTrial(fx("window"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-window-mpole", "[ff][fep]") { runTrial(fx("window-mpole"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-charge", "[ff][fep]") { runTrial(fx("charge"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-hippo", "[ff][fep]") { runTrial(fx("hippo"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-aplus", "[ff][fep]") { runTrial(fx("aplus"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-aplus-cross", "[ff][fep][cflux]") { runTrial(fx("aplus-cross"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-dispann", "[ff][fep][disp]") { runTrial(fx("dispann"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-dispcorr", "[ff][fep][disp]") { runTrial(fx("dispcorr"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-expol", "[ff][fep][expol]") { runTrial(fx("expol"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-expol-dt", "[ff][fep][expol]") { runTrial(fx("expol-dt"), kRcDyn); }
TEST_CASE("FEP-trial-dyn-expol-box", "[ff][fep][expol]") { runTrial(fx("expol-box"), kRcDyn); }

// An annihilated ion leaves no exchange polarization behind, in single and dual
// topology: the four waters alone.
TEST_CASE("FEP-end-expol", "[ff][fep][expol]")
{
   runEnd(fx("expol"), {{"fep/expolenv.key", "fep/expwat4.xyz"}}, testGetEps(1.0e-4, 1.0e-6));
}
TEST_CASE("FEP-end-expol-dt", "[ff][fep][expol]")
{
   runEnd(fx("expol-dt"), {{"fep/expolenv.key", "fep/expwat4.xyz"}}, testGetEps(1.0e-4, 1.0e-6));
}
TEST_CASE("FEP-trial-dyn-dispdec", "[ff][fep][disp]") { runTrial(fx("dispdec"), kRcDyn); }

// An annihilated ligand leaves the environment alone, and a decoupled one the
// environment and the ligand each alone, Ewald dispersion included.
TEST_CASE("FEP-end-dispann", "[ff][fep][disp]")
{
   runEnd(fx("dispann"), {{"fep/dispenv.key", "fep/dmso_env.xyz"}}, testGetEps(1.0e-4, 1.0e-6));
}
TEST_CASE("FEP-end-dispdec", "[ff][fep][disp]")
{
   runEnd(fx("dispdec"), {{"fep/dispenv.key", "fep/dmso_env.xyz"}, {"fep/dispenv.key", "fep/dmso_lig.xyz"}},
      testGetEps(1.0e-4, 1.0e-6));
}
TEST_CASE("FEP-linear-dispann", "[ff][fep][disp]") { runLinear("fep/displin_ann.key", 0.5, testGetEps(1.0e-4, 1.0e-6)); }
TEST_CASE("FEP-linear-dispdec", "[ff][fep][disp]") { runLinear("fep/displin_dec.key", 0.5, testGetEps(1.0e-4, 1.0e-6)); }

TEST_CASE("FEP-state-ast-ewald-ion", "[ff][fep]") { runState(fx("ast-ewald-ion")); }
TEST_CASE("FEP-state-adt-ewald", "[ff][fep]") { runState(fx("adt-ewald")); }
TEST_CASE("FEP-state-rels-dt", "[ff][fep]") { runState(fx("rels-dt")); }
TEST_CASE("FEP-state-rels-st", "[ff][fep]") { runState(fx("rels-st")); }
TEST_CASE("FEP-state-window", "[ff][fep]") { runState(fx("window")); }
TEST_CASE("FEP-state-charge", "[ff][fep]") { runState(fx("charge")); }
TEST_CASE("FEP-state-hippo", "[ff][fep]") { runState(fx("hippo")); }
TEST_CASE("FEP-state-aplus", "[ff][fep]") { runState(fx("aplus")); }
TEST_CASE("FEP-state-expol", "[ff][fep][expol]") { runState(fx("expol")); }
TEST_CASE("FEP-state-expol-dt", "[ff][fep][expol]") { runState(fx("expol-dt")); }
TEST_CASE("FEP-state-expol-box", "[ff][fep][expol]") { runState(fx("expol-box")); }

TEST_CASE("FEP-predict", "[ff][fep]")
{
   const Fixture& f = fx("window");
   const double trial[] = {0.5, 0.0};
   double e[2], en;

   // The Fortran reader turns the predictor the key file asks for off, since
   // its own trial energies would enter the history. The trial solves here
   // bypass the predictor, so the dynamics keeps it.
   {
      Setup s(f, kMd, std::string(f.native) + "\nintegrator verlet\n", false);
      testMdInit(298., 0.);
      REQUIRE(dlmda::use_fep);
      REQUIRE_FALSE(uprior::use_pred);
      s.session.init();
      REQUIRE(use_fep);
      REQUIRE(polpred == UPred::ASPC);
      REQUIRE(maxualt == 16);

      // fill the history, as a run of dynamics steps would
      std::vector<pos_prec> xb(n);
      for (int step = 0; step < 19; ++step) {
         darray::copyout(g::q0, n, xb.data(), xpos);
         waitFor(g::q0);
         for (int i = 0; i < n; ++i)
            xb[i] += 0.01 * ((i + step) % 3 - 1);
         darray::copyin(g::q0, n, xpos, xb.data());
         waitFor(g::q0);
         copyPosToXyz(true);
         energy(calc::v1);
      }
      REQUIRE(nualt >= maxualt);

      // the trial and native solves leave no trace in it
      const State before;
      REQUIRE(before.slots.size() == 16);
      efepTrial(2, trial, e, en);
      requireSame(State(), before);

      // and the solves outside a trial call still add to it
      energy(calc::v1);
      REQUIRE(nualt == before.nualt + 1);
      s.session.end();
   }

   // Dual topology keeps a history for each endpoint, and the trial and native
   // solves leave both as they were.
   {
      Setup s(f, kMd, std::string(f.native) + "\npol-dualtopo\nintegrator verlet\n", false);
      testMdInit(298., 0.);
      s.session.init();
      REQUIRE(use_fep);
      REQUIRE(use_epdt);
      REQUIRE(polpred == UPred::ASPC);
      REQUIRE(induceHistoryUsable());

      std::vector<pos_prec> xb(n);
      for (int step = 0; step < 19; ++step) {
         darray::copyout(g::q0, n, xb.data(), xpos);
         waitFor(g::q0);
         for (int i = 0; i < n; ++i)
            xb[i] += 0.01 * ((i + step) % 3 - 1);
         darray::copyin(g::q0, n, xpos, xb.data());
         waitFor(g::q0);
         copyPosToXyz(true);
         energy(calc::v1);
      }
      // an endpoint the native lambda gives no weight is not solved, and a
      // solve there would start over rather than extrapolate a stale history
      DtPass pass[2];
      dtPasses(pass);
      REQUIRE((pass[0].wa == 0) != (pass[1].wa == 0));
      std::vector<State> before;
      int len[2];
      for (int k : {0, 1}) {
         CAPTURE(k);
         len[k] = induceLoadHistory(pass[k].mask);
         before.emplace_back();
         if (pass[k].wa != 0)
            REQUIRE(len[k] >= maxualt);
         else
            REQUIRE(len[k] == 0);
      }

      efepTrial(2, trial, e, en);
      for (int k : {0, 1}) {
         CAPTURE(k);
         REQUIRE(induceLoadHistory(pass[k].mask) == len[k]);
         requireSame(State(), before[k]);
      }
      s.session.end();
   }

   // Without free energy perturbation the choice is the reader's alone.
   {
      Setup s(f, kMd, std::string(f.fresh) + "\nintegrator verlet\n", false);
      testMdInit(298., 0.);
      REQUIRE_FALSE(dlmda::use_fep);
      REQUIRE(uprior::use_pred);
      uprior::use_pred = 0;
      s.session.init();
      REQUIRE(polpred == UPred::NONE);
      s.session.end();
   }
}

TEST_CASE("FEP-sample", "[ff][fep]")
{
   const Fixture& f = fx("window");
   const double eps = testGetEps(f.eps, 1.0e-6);

   Setup s(f, kRc, f.native);
   FepFiles files("fepsave_tmp");

   // the mode owns the main lambda without the lambda derivative
   termsFep();
   REQUIRE(nlmdawin == 3);
   REQUIRE(fepintv == 4);
   REQUIRE(lmdawinratio == 0.5);
   REQUIRE(lambda == 1.0);

   // Sixty steps give three windows of twenty steps, each with ten production
   // steps and so two samples; the end windows have one neighbor and the
   // middle window has two.
   init_fepdyn(60);
   std::string fepfile = files.claim();
   REQUIRE(fepfile.size() > 0);
   REQUIRE(nfeptot == 14);
   REQUIRE(fepCountRows(fepfile) == 0); // header only
   REQUIRE_FALSE(fepHasLine(fepfile, "Simulation Pressure"));
   for (int istep = 1; istep <= 60; ++istep) {
      lmdaWinDyn(istep);
      lmdaWinStep(istep);
   }
   REQUIRE(nfep == 14);

   // the first window samples at steps 14 and 18, native first
   REQUIRE(fepstep[0] == 14);
   REQUIRE(feplmda[0] == 1.0);
   REQUIRE(feptrial[0] == 1.0);
   REQUIRE(fepstep[1] == 14);
   REQUIRE(feplmda[1] == 1.0);
   REQUIRE(feptrial[1] == 0.5);
   REQUIRE(fepstep[2] == 18);
   // the middle window lists the previous window, then the next
   REQUIRE(fepstep[4] == 34);
   REQUIRE(feplmda[4] == 0.5);
   REQUIRE(feptrial[4] == 0.5);
   REQUIRE(feptrial[5] == 1.0);
   REQUIRE(feptrial[6] == 0.0);
   REQUIRE(feplmda[6] == 0.5);
   // the last window samples at steps 54 and 58
   REQUIRE(fepstep[12] == 58);
   REQUIRE(feplmda[12] == 0.0);
   REQUIRE(feptrial[13] == 0.5);

   // The structure never moved, so a trial energy found from one window is the
   // native energy found in the neighboring window.
   COMPARE_REALS(fepene[2], fepene[0], eps);
   COMPARE_REALS(fepene[1], fepene[4], eps);
   COMPARE_REALS(fepene[5], fepene[0], eps);
   COMPARE_REALS(fepene[6], fepene[10], eps);

   // nor did the box, whose volume each record keeps
   REQUIRE(boxVolume() > 0);
   for (int i = 0; i < nfep; ++i)
      REQUIRE(fepvol[i] == boxVolume());

   // Write the records in two parts, and check that a call with nothing new
   // adds no rows.
   nfep = 5;
   mdsaveLmdaFinal(20);
   REQUIRE(fepCountRows(fepfile) == 5);
   nfep = 14;
   mdsaveLmdaFinal(60);
   mdsaveLmdaFinal(60);
   REQUIRE(fepCountRows(fepfile) == 14);
   REQUIRE(fep::nfepsave == 14);
   int istep = 0;
   double simlmda = -1, trial = -1, ene = 0, vol = 0;
   REQUIRE(fepRow(fepfile, 6, istep, simlmda, trial, ene, &vol));
   REQUIRE(istep == 34);
   COMPARE_REALS(simlmda, 0.5, 1.0e-8);
   COMPARE_REALS(trial, 1.0, 1.0e-8);
   COMPARE_REALS(ene, fepene[5], 1.0e-8);
   REQUIRE(std::fabs(vol - fepvol[5]) <= 1.0e-4);

   // an isobaric run gives its pressure in the header
   const int oldisobaric = bath::isobaric;
   const double oldatmsph = bath::atmsph;
   bath::isobaric = 1;
   bath::atmsph = 1.0;
   init_fepdyn(60);
   fepfile = files.claim();
   bath::isobaric = oldisobaric;
   bath::atmsph = oldatmsph;
   REQUIRE(fepHasLine(fepfile, format(" Simulation Pressure%16s%8.2f", "", 1.0)));

   // An ascending schedule lists its neighbors the same way.
   dlmda::lmdawinlist[0] = 0.0;
   dlmda::lmdawinlist[2] = 1.0;
   lmdawinlist[0] = 0.0;
   lmdawinlist[2] = 1.0;
   init_fepdyn(60);
   files.claim();
   REQUIRE(lambda == 0.0);
   for (int i = 1; i <= 60; ++i) {
      lmdaWinDyn(i);
      lmdaWinStep(i);
   }
   REQUIRE(nfep == 14);
   REQUIRE(feplmda[0] == 0.0);
   REQUIRE(feptrial[1] == 0.5);
   REQUIRE(feptrial[5] == 0.0);
   REQUIRE(feptrial[6] == 1.0);
   REQUIRE(feplmda[13] == 1.0);

   s.session.end();
}

TEST_CASE("FEP-dyn", "[ff][fep]")
{
   // The trajectory is the one that the same run takes when no trial energy
   // is found at all.
   DynOpts o;
   const DynEnd on = runDyn(o);
   o.sample = false;
   const DynEnd off = runDyn(o);
   REQUIRE(on.nfep > 0);
   REQUIRE(off.nfep == 0);
   REQUIRE_FALSE(sameBits(off.x, off.x0));
   REQUIRE(sameEnd(on, off));
}

TEST_CASE("FEP-dyn-midwindow", "[ff][fep]")
{
   // The same, with steps left to take in a window once it has been sampled.
   DynOpts o;
   o.extra = kTwice;
   const DynEnd on = runDyn(o);
   o.sample = false;
   const DynEnd off = runDyn(o);
   REQUIRE(on.nfep == 14);
   REQUIRE(on.fepstep[0] == 6);
   REQUIRE(on.fepstep[2] == 8);
   REQUIRE(off.nfep == 0);
   REQUIRE(sameEnd(on, off));
}

TEST_CASE("FEP-dyn-rescaled", "[ff][fep]")
{
   // A barostat that rescales the coordinates after the force call of a step
   // leaves the neighbor lists behind the structure. The energies recorded are
   // those of the structure the step ended at, which is the one saved.
   DynOpts o;
   o.baro = BarostatEnum::BERENDSEN;
   o.check = true;
   const DynEnd on = runDyn(o);
   REQUIRE(on.nfep > 0);
   REQUIRE_FALSE(sameBits(on.x, on.x0));
   REQUIRE(on.box.lvec1.x != on.box0.lvec1.x);
   // and the volumes recorded follow the box
   REQUIRE(on.fepvol.front() != on.fepvol.back());
}

TEST_CASE("FEP-dyn-propagate", "[ff][fep]")
{
   // The integrators leave the sampling of the window to mdPropagate(). Its
   // run finds the trial energies at the steps the loop of the other cases
   // does, and ends where that loop ends.
   DynOpts o;
   const DynEnd loop = runDyn(o);
   o.propagate = true;
   const DynEnd prop = runDyn(o);
   REQUIRE(loop.nfep > 0);
   REQUIRE(prop.nfep == loop.nfep);
   REQUIRE(prop.fepstep == loop.fepstep);
   REQUIRE(prop.feplmda == loop.feplmda);
   REQUIRE(prop.feptrial == loop.feptrial);
   REQUIRE_FALSE(sameBits(prop.x, prop.x0));
   REQUIRE(sameEnd(prop, loop));
}

TEST_CASE("FEP-dyn-save-lambda", "[ff][fep]")
{
   // A saved frame prints the Fortran main lambda, which the save takes from
   // the lambda of the window the frame is in. Saves on the last step of each
   // window leave it at the lambda of the last window, where without them it
   // keeps the lambda the setup gave it.
   DynOpts o;
   o.propagate = true;
   o.iwrite = 8;
   const DynEnd end = runDyn(o);
   REQUIRE(end.lmdastart != end.lmdalast);
   REQUIRE(end.lmdasaved == end.lmdalast);
}

TEST_CASE("FEP-dyn-dual", "[ff][fep]")
{
   // Dual topology polarization, whose trial energies weigh two solves apiece,
   // leaves the trajectory alone as the single topology does, predictor
   // histories of both endpoints included.
   DynOpts o;
   o.extra = std::string("\npol-dualtopo\n") + kTwice;
   o.pred = UPred::ASPC;
   o.terms = []() {
      REQUIRE(use_fep);
      REQUIRE(use_epdt);
   };
   const DynEnd on = runDyn(o);
   o.sample = false;
   const DynEnd off = runDyn(o);
   REQUIRE(on.nfep == 14);
   REQUIRE(off.nfep == 0);
   REQUIRE_FALSE(sameBits(off.x, off.x0));
   REQUIRE(sameEnd(on, off));
}

TEST_CASE("FEP-dyn-staged", "[ff][fep]")
{
   // A window in each leg of the staged relative schedule, sampled twice, so
   // that the run steps from leg to leg and each sample has a trial lambda in
   // another leg.
   // The barostat moves the structure after the force call of a step, which
   // tells the native energy found for the sample from the one the step left.
   DynOpts o;
   o.fixture = "rels-dt";
   o.extra = "\nlambda-mode fep\nlambda-window 0.85\nlambda-window 0.5\nlambda-window 0.15\n"
             "lambda-eqratio 0.5\nlambda-interval 2\n";
   o.baro = BarostatEnum::BERENDSEN;
   o.check = true;
   o.pred = UPred::NONE;
   o.terms = []() {
      REQUIRE(use_fep);
      REQUIRE(use_rel);
      REQUIRE(nlmdawin == 3);
   };
   const DynEnd on = runDyn(o);

   // each sample records its native energy, then those of its neighbors
   REQUIRE(on.nfep == 14);
   REQUIRE(on.legs
      == std::vector<RelStage>{RelStage::LIG1, RelStage::LIG1, RelStage::VDWM, RelStage::VDWM, RelStage::LIG2,
            RelStage::LIG2});
   const double lref[] = {0.85, 0.85, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.15, 0.15, 0.15, 0.15};
   const double tref[] = {0.85, 0.5, 0.85, 0.5, 0.5, 0.85, 0.15, 0.5, 0.85, 0.15, 0.15, 0.5, 0.15, 0.5};
   for (int i = 0; i < 14; ++i) {
      COMPARE_REALS(on.feplmda[i], lref[i], 1.0e-8);
      COMPARE_REALS(on.feptrial[i], tref[i], 1.0e-8);
   }
   REQUIRE_FALSE(sameBits(on.x, on.x0));
   REQUIRE(on.box.lvec1.x != on.box0.lvec1.x);
}

TEST_CASE("FEP-dyn-montecarlo", "[ff][fep]")
{
   // A Monte Carlo volume trial uses the predictor history and keeps what it
   // solved when the move is accepted; the trial energies of the same step
   // bypass the history. One after the other, at every step, they leave the
   // run where it ends with no trial energy found at all.
   DynOpts o;
   o.extra = std::string("\nvolume-trial 1\nvolume-move 40.0\n") + kTwice;
   o.baro = BarostatEnum::MONTECARLO;
   o.sample = false;
   const DynEnd off = runDyn(o);
   // the volume moves are drawn alike from one session to the next
   const DynEnd again = runDyn(o);
   REQUIRE(sameEnd(again, off));
   REQUIRE(sameBox(again.box, off.box));

   o.sample = true;
   o.check = true;
   const DynEnd on = runDyn(o);
   REQUIRE(on.nfep > 0);
   // Moves were accepted, and rejected, and one of those accepted was on a
   // step that was sampled with more of the run to come.
   CAPTURE(on.nmoved, on.nmovedsample);
   REQUIRE(on.nmoved > 0);
   REQUIRE(on.nmoved < 24);
   REQUIRE(on.nmovedsample > 0);
   REQUIRE(on.nmoved == off.nmoved);
   REQUIRE_FALSE(sameBox(off.box, off.box0));
   REQUIRE(sameEnd(on, off));
   REQUIRE(sameBox(on.box, off.box));
}

#endif
