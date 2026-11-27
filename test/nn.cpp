#include "ff/dlmda.h"
#include "ff/efep.h"
#include "ff/energy.h"
#include "ff/ennintermol.h"
#include "ff/evalence.h"
#include "tool/xtesthelper.h"

#include <fstream>
#include <sstream>
#include <utility>

#include "test.h"
#include "testrt.h"

using namespace tinker;

#if TINKER_GPULANG_CUDA

TEST_CASE("NNMET-CU-WATER", "[ff][nn][ennmet][cu-water][L1]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", "nn_6o.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nn_6o.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   const char* xn = "nn_6o.xyz";
   const char* kn = "nn_6o.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0005, 0.0001);
   const double eps_g = testGetEps(0.0005, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/nn_6o.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(nennmet, ref_c);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}

TEST_CASE("NNVAL-ALATET", "[ff][nn][ennval][alatet][L1]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/nn/alatet_nnval.xyz", "alatet_nnval.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/nn/alatet_nnval.key", "alatet_nnval.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/nn/amoebann_v3.prm", "amoebann_v3.prm");
   const char* xn = "alatet_nnval.xyz";
   const char* kn = "alatet_nnval.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   // The reference is a single precision result: a mixed build matches its
   // printed gradient to the last digit, and a double one differs by up to 9e-4.
   const double eps_e = testGetEps(0.0005, 0.0001);
   const double eps_g = testGetEps(0.0005, 0.002);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/alatet_nnval.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(nennval, ref_c);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}


namespace {
const char* const kNNXyz = "nn_6o.xyz";
const char* const kNNKey = "nn_6o.key";
const int kNNFlag = calc::xyz | calc::vmask;

// The Cu(H2O)6 system with the atoms in each closed range removed, renumbered
// with its connectivity, as the state in which those atoms are absent.
std::string nnXyzWithout(std::vector<std::pair<int, int>> drop)
{
   std::ifstream in(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz");
   std::string line, box;
   std::getline(in, line);
   const int n = std::stoi(line);
   std::getline(in, box);
   std::vector<std::vector<std::string>> atoms(n);
   for (int i = 0; i < n; ++i) {
      std::getline(in, line);
      std::istringstream ss(line);
      std::string t;
      while (ss >> t)
         atoms[i].push_back(t);
   }
   auto dropped = [&](int i) {
      for (auto& r : drop)
         if (i >= r.first and i <= r.second)
            return true;
      return false;
   };
   std::vector<int> id(n + 1, 0);
   int m = 0;
   for (int i = 1; i <= n; ++i)
      if (not dropped(i))
         id[i] = ++m;
   std::ostringstream out;
   out << m << "\n" << box << "\n";
   for (int i = 1; i <= n; ++i) {
      if (dropped(i))
         continue;
      const auto& t = atoms[i - 1];
      out << id[i] << " " << t[1] << " " << t[2] << " " << t[3] << " " << t[4] << " " << t[5];
      for (size_t k = 6; k < t.size(); ++k)
         out << " " << id[std::stoi(t[k])];
      out << "\n";
   }
   return out.str();
}

// Energy, gradient and virial of one neural network state; a gradient from a
// system with atoms removed is spread back onto all 19 atoms, with zeros on
// the removed ones.
struct NNState
{
   double e;
   std::vector<double> g;
   double v[9];
};

NNState nnState(const char* xyz, std::vector<std::pair<int, int>> drop = {})
{
   const char* argv[] = {"dummy", xyz, "-k", kNNKey};
   rc_flag = kNNFlag;
   TestSession session(4, argv);
   session.init();
   energy(calc::v1);
   NNState s;
   s.e = esum;
   std::vector<double> g;
   copyGradientFlat(calc::grad, g);
   for (int k = 0; k < 9; ++k)
      s.v[k] = vir[k];
   session.end();
   s.g.assign(3 * 19, 0.0);
   int m = 0;
   for (int i = 1; i <= 19; ++i) {
      bool gone = false;
      for (auto& r : drop)
         gone = gone or (i >= r.first and i <= r.second);
      if (gone)
         continue;
      for (int j = 0; j < 3; ++j)
         s.g[3 * (i - 1) + j] = g[3 * m + j];
      ++m;
   }
   return s;
}

// The mix (1-w)*off + w*on with slope w1 and curvature w2 must give the energy,
// gradient, virial and lambda derivatives of the current session.
void nnRequireMix(const NNState& off, const NNState& on, double w, double w1, double w2, bool analysis)
{
   const double eps_e = 1.0e-4, eps_g = 5.0e-4, eps_v = 5.0e-4;
   COMPARE_REALS(esum, (1 - w) * off.e + w * on.e, eps_e);
   COMPARE_REALS(dedl, w1 * (on.e - off.e), eps_e);
   COMPARE_REALS(d2edl2, w2 * (on.e - off.e), eps_e);
   if (analysis) {
      COMPARE_REALS(denndl, dedl, eps_e);
      COMPARE_REALS(d2enndl2, d2edl2, eps_e);
   }
   std::vector<double> g, f, mix(on.g.size()), slope(on.g.size());
   copyGradientFlat(calc::grad, g);
   copyGradientFlat(calc::grad, f, dfdlx, dfdly, dfdlz);
   for (size_t k = 0; k < mix.size(); ++k) {
      mix[k] = (1 - w) * off.g[k] + w * on.g[k];
      slope[k] = w1 * (on.g[k] - off.g[k]);
   }
   testRequireNear(g, mix, eps_g);
   testRequireNear(f, slope, eps_g);
   for (int k = 0; k < 9; ++k) {
      INFO(k);
      REQUIRE(vir[k] == Approx((1 - w) * off.v[k] + w * on.v[k]).margin(eps_v));
      REQUIRE(dvirdl[k] == Approx(w1 * (on.v[k] - off.v[k])).margin(eps_v));
   }
}
}

TEST_CASE("NNMET-lambda-ligand", "[ff][nn][ennmet][lambda][L1]")
{
   // One coordinating water is the ligand: the uncoupled state is the system
   // without it, the coupled one the whole system, mixed at the neural network
   // lambda w = lambda^2 of an EXP 2 map, in and out of analysis.
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fk(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile fd("", "nn_5o.xyz", nnXyzWithout({{2, 4}}));
   TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key",
      "ligand -2 4\nlambda 0.4\nlambda-deriv2\nnn-lmda-map exp\nnn-lmda-exp 2\n");
   const NNState on = nnState(kNNXyz);
   const NNState off = nnState("nn_5o.xyz", {{2, 4}});
   REQUIRE(on.e != Approx(off.e));

   for (bool analysis : {true, false}) {
      INFO(analysis);
      const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
      rc_flag = analysis ? kNNFlag : (kNNFlag & ~calc::analyz);
      TestSession session(4, argv);
      session.init();
      REQUIRE(use_nndlmda);
      for (double l : {0.4, 1.0, 0.0}) {
         INFO(l);
         lambda = l;
         energy(calc::v1);
         nnRequireMix(off, on, l * l, 2 * l, 2, analysis);
         if (not analysis) {
            // a step that keeps the energy only for its lambda derivatives
            const double full = dedl;
            energy(calc::v1 | calc::dlmda_only);
            COMPARE_REALS(dedl, full, 1.0e-6);
         }
      }

      // free energy perturbation trials follow the neural network lambda and
      // leave the native state as it was
      lambda = 0.4;
      energy(calc::v0);
      const double ltrial[3] = {0.0, 0.5, 1.0};
      double etrial[3], enative;
      efepTrial(3, ltrial, etrial, enative);
      for (int k = 0; k < 3; ++k) {
         const double w = ltrial[k] * ltrial[k];
         COMPARE_REALS(etrial[k], (1 - w) * off.e + w * on.e, 1.0e-4);
      }
      COMPARE_REALS(enative, 0.84 * off.e + 0.16 * on.e, 1.0e-4);
      REQUIRE(nnlam == Approx(0.16));
      session.end();
   }
}

TEST_CASE("NNMET-lambda-fd", "[ff][nn][ennmet][lambda][L2]")
{
   // testlmda, analytical against numerical; the energy is quadratic in lambda
   // on an EXP 2 map, so a large step costs no truncation error
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey,
      "ligand -2 4\nlambda 0.4\nlambda-deriv2\nnn-lmda-map exp\nnn-lmda-exp 2\n");
   const char* argv[] = {"dummy", kNNXyz, "-k", kNNKey};
   TestSession session(4, argv);
   FdTestOptions opts;
   opts.eps = 0.05;
   rc_flag = testlmdaFlags(opts);
   session.init();

   const auto r = testlmdaEvaluate(opts);
   REQUIRE(r.dedl[0] != 0);
   COMPARE_REALS(r.ndedl[0], r.dedl[0], 1.0e-3);
   // the neural network column carries all of it, the other terms none
   const int nn = nlmdacol - 1; // the last column
   COMPARE_REALS(r.dedl[nn], r.dedl[0], 1.0e-6);
   COMPARE_REALS(r.ndedl[nn], r.ndedl[0], 1.0e-6);
   COMPARE_REALS(r.d2edl2[nn], r.d2edl2[0], 1.0e-6);
   for (int k = 1; k < nn; ++k) {
      REQUIRE(r.dedl[k] == 0);
      REQUIRE(r.ndedl[k] == 0);
   }
   COMPARE_REALS(r.nd2edl2[0], r.d2edl2[0], 1.0e-3);
   testRequireNear(r.ndfdl, r.dfdl, 1.0e-3);
   for (int k = 0; k < 9; ++k) {
      INFO(k);
      REQUIRE(r.ndvirdl[k] == Approx(r.dvirdl[k]).margin(1.0e-3));
   }
   session.end();
}

TEST_CASE("NNMET-lambda-annihilate", "[ff][nn][ennmet][lambda][L1]")
{
   // The Cu ion is the ligand; annihilated, its uncoupled state has no center
   // at all, so the energy is w times that of the whole system.
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fk(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key",
      "ligand -1 1\nvdw-annihilate\nlambda 0.4\nlambda-deriv2\nnn-lmda-map exp\nnn-lmda-exp 2\n");
   const NNState on = nnState(kNNXyz);
   NNState none = on;
   none.e = 0;
   std::fill(none.g.begin(), none.g.end(), 0.0);
   std::fill(none.v, none.v + 9, 0.0);

   const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
   rc_flag = kNNFlag;
   TestSession session(4, argv);
   session.init();
   energy(calc::v1);
   nnRequireMix(none, on, 0.16, 0.8, 2, true);
   session.end();
}

TEST_CASE("NNMET-lambda-relative", "[ff][nn][ennmet][lambda][L1]")
{
   // Two waters are the two ligands of a staged relative run: LIGNN2 removes
   // the coupling of ligand 2 below the morph and LIGNN1 restores that of
   // ligand 1 above it, and the two ligands never see each other.
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fk(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile f1("", "nn_on1.xyz", nnXyzWithout({{8, 10}}));
   TestFile f2("", "nn_on2.xyz", nnXyzWithout({{5, 7}}));
   TestFile f0("", "nn_off.xyz", nnXyzWithout({{5, 10}}));
   TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key",
      "ligand1 -5 7\nligand2 -8 10\nlambda 0.2\nlambda-deriv2\n"
      "rel-stage lignn2 0.0 0.4\nrel-stage vdwm 0.4 0.6\nrel-stage lignn1 0.6 1.0\n");
   const NNState on1 = nnState("nn_on1.xyz", {{8, 10}});
   const NNState on2 = nnState("nn_on2.xyz", {{5, 7}});
   const NNState off = nnState("nn_off.xyz", {{5, 10}});

   const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
   rc_flag = kNNFlag;
   TestSession session(4, argv);
   session.init();
   REQUIRE(use_rel);
   REQUIRE(use_nndlmda);

   // the middle of each neural network leg, on the quintic map
   lambda = 0.2;
   energy(calc::v1);
   REQUIRE(relstage == RelStage::LIG2);
   nnRequireMix(off, on2, 0.5, -4.6875, 0, true);
   lambda = 0.8;
   energy(calc::v1);
   REQUIRE(relstage == RelStage::LIG1);
   nnRequireMix(off, on1, 0.5, 4.6875, 0, true);
   // no ligand is coupled within the morph
   lambda = 0.5;
   energy(calc::v1);
   REQUIRE(relstage == RelStage::VDWM);
   nnRequireMix(off, on1, 0.0, 0.0, 0, true);
   session.end();
}



TEST_CASE("NNMET-lambda-no-ligand", "[ff][nn][ennmet][lambda][L1]")
{
   // NN-LAMBDA and its map couple the mutated atoms to the metal term, so with
   // none they change nothing, as ELE-LAMBDA without mutated atoms
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestReference r(TINKER9_DIRSTR "/test/ref/nn_6o.txt");
   const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
   for (const char* extra : {"nn-lambda 0.5\n", "lambda 0.5\nnn-lmda-map exp\n"}) {
      INFO(extra);
      TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key", extra);
      rc_flag = kNNFlag;
      TestSession session(4, argv);
      session.init();
      energy(calc::v0);
      COMPARE_REALS(esum, r.getEnergy(), testGetEps(0.0005, 0.0001));
      session.end();
   }
}

TEST_CASE("NNMET-group-once", "[ff][nn][ennmet][L1]")
{
   // an NNTERM line repeated in any case, here with a trailing comment, names
   // its group once, so each atom of the group is one center
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key", "NNTERM METAL 1 # copper\n");
   TestReference r(TINKER9_DIRSTR "/test/ref/nn_6o.txt");
   const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
   rc_flag = kNNFlag;
   TestSession session(4, argv);
   session.init();
   energy(calc::v3);
   COMPARE_INTS(nennmet, r.getCount());
   COMPARE_REALS(esum, r.getEnergy(), testGetEps(0.0005, 0.0001));
   session.end();
}


TEST_CASE("NNMET-lambda-ligand-center", "[ff][nn][ennmet][lambda][L1]")
{
   // The Cu ion and one coordinating water are the ligand, so the ligand holds
   // the center. Decoupled, the uncoupled state keeps the pair within the
   // ligand and is the system of those four atoms alone; annihilated, it has
   // no center at all.
   TestLmdaFlagReset lmdaReset;
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fk(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   TestFile f4("", "nn_lig.xyz", nnXyzWithout({{5, 19}}));
   const NNState on = nnState(kNNXyz);
   const NNState lig = nnState("nn_lig.xyz", {{5, 19}});
   REQUIRE(lig.e != Approx(0.0));
   REQUIRE(lig.e != Approx(on.e));
   NNState none = on;
   none.e = 0;
   std::fill(none.g.begin(), none.g.end(), 0.0);
   std::fill(none.v, none.v + 9, 0.0);

   const std::string key = "ligand -1 4\nlambda 0.4\nlambda-deriv2\nnn-lmda-map exp\nnn-lmda-exp 2\n";
   for (bool annihilate : {false, true}) {
      INFO(annihilate);
      TestFile fl(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nnl.key", key + (annihilate ? "vdw-annihilate\n" : ""));
      const char* argv[] = {"dummy", kNNXyz, "-k", "nnl.key"};
      rc_flag = kNNFlag;
      TestSession session(4, argv);
      session.init();
      energy(calc::v1);
      nnRequireMix(annihilate ? none : lig, on, 0.16, 0.8, 2, true);
      session.end();
   }
}

TEST_CASE("NNMET-lambda-prm-keyword", "[ff][nn][ennmet][lambda][L1]")
{
   // NN-LAMBDA is a mutation keyword of the key file; in the parameter file it
   // would be ignored, so the setup stops
   TestFile fx(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", kNNXyz);
   TestFile fk(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", kNNKey);
   TestFile fp(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm", "nn-lambda 0.5\n");
   const char* argv[] = {"dummy", kNNXyz, "-k", kNNKey};
   rc_flag = kNNFlag;
   TestSession session(4, argv);
   REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring("Belong in the Key File"));
}

#endif
