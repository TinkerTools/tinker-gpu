#include "ff/dlmda.h"
#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/nblist.h"
#include "test.h"
#include "testrt.h"
#include "tool/platform.h"
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>

using namespace tinker;

TEST_CASE("Chglj-Trpcage", "[ff][echarge][evdw][echglj][lj][trpcage]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;

   const char* kn = "test_chglj.key";
   std::string k0 = R"**(
bondterm     none
angleterm    none
impropterm   none
torsionterm  none
)**";
   const char* xn = "test_chglj.xyz";

   const double eps_e = 0.0001;
   const double eps_g = 0.0005;
   const double eps_v = 0.001;
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   SECTION("  - echglj -- no pbc, no cutoff")
   {
      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k0);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.1.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      session.end();
   }

   SECTION("  - echglj -- pbc, cutoff")
   {
      std::string k1 = k0 + R"**(
ewald
cutoff 9.0
neighbor-list
list-buffer 0.5
a-axis  30
b-axis  25
c-axis  20
vdw-correction
)**";

      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k1);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.2.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      session.end();
   }

   // The fused kernel mixes the square roots of the radii here. Each radius is
   // rounded once per atom type and LJ amplifies that about twelvefold, so the
   // mixed-precision diagonal virial wanders by about 2e-3 around the reference.
   SECTION("  - echglj -- pbc, cutoff, geometric radius rule")
   {
      const double eps_v = 0.005;

      std::string k1 = k0 + R"**(
ewald
cutoff 9.0
neighbor-list
list-buffer 0.5
a-axis  30
b-axis  25
c-axis  20
vdw-correction
radiusrule GEOMETRIC
)**";

      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k1);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.3.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();
      // the fused kernel needs the spatial list of the GPU builds
      if (clistVersion() & Nbl::SPATIAL)
         REQUIRE(echgljAvailable());

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      session.end();
   }

   // Fortran mixes the HARMONIC (and MMFF94) epsilons into its tables, which
   // evdw() reads, but the fused kernel cannot take them. Following the
   // geometric sections, this also checks that no rule of a session before it
   // is kept: a Debug build used to abort here, and a Release one to run the
   // fused kernel with the geometric rule left over.
   SECTION("  - echarge and evdw -- pbc, cutoff, harmonic epsilon rule")
   {
      std::string k1 = k0 + R"**(
ewald
cutoff 9.0
neighbor-list
list-buffer 0.5
a-axis  30
b-axis  25
c-axis  20
vdw-correction
epsilonrule HARMONIC
)**";

      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k1);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.5.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();
      REQUIRE_FALSE(echgljAvailable());

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      session.end();
      // freed with the session, so that none of its charges outlive it
      REQUIRE(pchg == nullptr);
   }

   // The fused kernel mixes geometric epsilons only, so an arithmetic epsilon
   // rule has to fall back to echarge() and evdw().
   SECTION("  - echarge and evdw -- pbc, cutoff, arithmetic epsilon rule")
   {
      std::string k1 = k0 + R"**(
ewald
cutoff 9.0
neighbor-list
list-buffer 0.5
a-axis  30
b-axis  25
c-axis  20
vdw-correction
epsilonrule ARITHMETIC
)**";

      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k1);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.4.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();
      REQUIRE_FALSE(echgljAvailable());

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      session.end();
   }

   // With the analysis buffers the fused kernel gives way to echarge() and
   // evdw(), but PME stays on its own stream. A repeated v1 call used to sum
   // the charge energy before the PME stream had finished.
   SECTION("  - echarge and evdw -- pbc, cutoff, analysis buffers")
   {
      rc_flag = calc::xyz | calc::vmask;

      std::string k1 = k0 + R"**(
ewald
cutoff 9.0
neighbor-list
list-buffer 0.5
a-axis  30
b-axis  25
c-axis  20
vdw-correction
)**";

      TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
      TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, k1);
      TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/chglj.2.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_g = r.getGradient();

      TestSession session(argc, argv);
      session.init();

      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);
      const double ref_ec = energy_elec;

      for (int rep = 0; rep < 4; ++rep) {
         energy(calc::v1);
         COMPARE_REALS(esum, ref_e, eps_e);
         COMPARE_REALS(energy_elec, ref_ec, eps_e);
         COMPARE_GRADIENT(ref_g, eps_g);
         COMPARE_VIR9(vir, ref_v, eps_v);
      }

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_REALS(energy_elec, ref_ec, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      COMPARE_VIR9(vir, ref_v, eps_v);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_REALS(energy_elec, ref_ec, eps_e);

      session.end();
   }
}

// Neutral group cutoffs and neighbors (kcharge.f) are refused rather than
// silently left out.
TEST_CASE("Chglj-NeutralGroups-Refused", "[ff][echarge][trpcage]")
{
   rc_flag = calc::xyz | calc::energy;

   const char* kn = "test_chglj.key";
   const char* xn = "test_chglj.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, "\ncutoff 9.0\nneutral-groups\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   TestSession session(argc, argv);
   REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring("NEUTRAL-GROUPS"));
}

// A keyword "CHARGE -i value" gives atom i its own partial charge (kcharge.f),
// so giving it to every atom of a type is the run of the type keyword.
TEST_CASE("Chglj-AtomChargeKeyword", "[ff][echarge][trpcage]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad;

   const char* kn = "test_chglj.key";
   const char* xn = "test_chglj.xyz";
   const std::string xyz = TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   // the carbonyl oxygens of charmm19, type 57 at -0.55
   std::string atomkeys;
   std::ifstream fin(xyz);
   std::string line;
   std::getline(fin, line);
   while (std::getline(fin, line)) {
      std::istringstream ss(line);
      int i, t;
      std::string name;
      double x, y, z;
      if (ss >> i >> name >> x >> y >> z >> t and t == 57)
         atomkeys += "\ncharge " + std::to_string(-i) + " -0.60";
   }
   REQUIRE(atomkeys.size() > 0);

   auto run = [&](const std::string& extra, double& e, std::vector<double> (&g)[3]) {
      TestFile fx(xyz, xn);
      TestFile fk(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kn, extra + "\n");
      TestFile fp(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");
      TestSession session(argc, argv);
      session.init();
      energy(calc::v4);
      e = esum;
      for (auto& v : g)
         v.resize(n);
      copyGradient(calc::grad, g[0].data(), g[1].data(), g[2].data());
      session.end();
   };

   double e0, et, ea;
   std::vector<double> g0[3], gt[3], ga[3];
   run("", e0, g0);
   run("\ncharge 57 -0.60", et, gt);
   run(atomkeys, ea, ga);
   // the same charges: the energies sum in fixed point, but OpenACC gradients
   // are not reproducible to the bit
   REQUIRE(std::fabs(et - e0) > 1);
   REQUIRE(std::fabs(ea - et) < 1e-6);
   for (int a = 0; a < 3; ++a) {
      REQUIRE(ga[a].size() == gt[a].size());
      for (size_t i = 0; i < ga[a].size(); ++i)
         REQUIRE(ga[a][i] == Approx(gt[a][i]).margin(1e-6));
   }
}

#if TINKER_GPULANG_CUDA
namespace {
struct ChgljCase
{
   const char* xyz; ///< Coordinates, under test/file.
   std::string key; ///< Key file, under test/file.
   const char* prm; ///< Parameter file under test/file, if any.
};

// The energy, and with lambda derivatives their values, by the fused kernel
// without analysis and by the separate charge and Lennard-Jones kernels under
// analysis.
TestLmdaSnapshot chgljLmdaEval(const ChgljCase& c, bool analyz)
{
   TestLmdaFlagReset lmdaReset;
   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyz = testBaseName(c.xyz), key = testBaseName(c.key);
   TestFile fx(files + c.xyz, xyz);
   TestFile fk(files + c.key, key);
   std::unique_ptr<TestFile> fp;
   if (c.prm)
      fp.reset(new TestFile(files + c.prm));

   const char* argv[] = {"dummy", xyz.c_str(), "-k", key.c_str()};
   TestSession session(4, argv);
   rc_flag = calc::xyz | calc::mass | calc::vmask;
   if (not analyz)
      rc_flag &= ~calc::analyz;
   session.init();
   REQUIRE(echgljAvailable());
   TestLmdaSnapshot r = testLmdaSnapshot();
   session.end();
   return r;
}
}

// The fused kernel with lambda derivatives against the separate kernels, single
// topology through a neighbor list, and the second ligand charging on the
// staged relative leg by double loop (testlmda.cpp cases 157 and 163).
TEST_CASE("Chglj-Lmda-FusedVsSeparate", "[ff][echglj][charge][lj]")
{
   // the fused and separate kernels sum in different orders
   for (const char* name : {"157_tip3_both_nlist_l06", "163_tip3_rels_lig2_l04"}) {
      INFO(name);
      const ChgljCase c = {"testlmda/tip3box.xyz", std::string("testlmda/") + name + ".key",
         "commit_350df099/amber99sb.prm"};
      testLmdaCompare(chgljLmdaEval(c, false), chgljLmdaEval(c, true), 1.0e-4, 1.0e-3, 1.0e-3);
   }
}

#endif

#if !TINKER_GPULANG_CUDA
// The lambda derivative versions are CUDA only. dlmdaData2 refuses them first,
// but the OpenACC kernels refuse them as well instead of taking no term at
// all; without a neighbor list every term here runs on OpenACC.
TEST_CASE("Chglj-AccRefusesLambdaVersions", "[ff][echarge][evdw][hippo]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   const char* kn = "test_chglj.key";
   const char* xn = "test_chglj.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;
   const std::string f = TINKER9_DIRSTR "/test/file/";
   struct Case
   {
      std::string xyz, key, prm, extra;
      void (*term)(int);
   };
   const std::string trp = f + "trpcage/trp_charmm.";
   const std::string ewald = "ewald\ncutoff 9.0\na-axis 30\nb-axis 25\nc-axis 20\n";
   const std::string hippo = "parameters water21\nchgflxterm none\n";
   const Case cases[] = {
      {trp + "xyz", trp + "key", f + "commit_11e84c69/charmm19.prm", "\nvdwterm none\n", echarge},
      {trp + "xyz", trp + "key", f + "commit_11e84c69/charmm19.prm", "\nvdwterm none\n" + ewald, echarge},
      {trp + "xyz", trp + "key", f + "commit_11e84c69/charmm19.prm", "\nchargeterm none\n", elj},
      {f + "nacl/nacl1.xyz", "", f + "commit_6fe8e913/amoeba09.prm", "parameters amoeba09\nvdwterm only\n", ehal},
      {f + "testlmda/water2.xyz", "", f + "commit_aad9340c/water21.prm", hippo + "repulsionterm only\n", erepel},
      {f + "testlmda/water2.xyz", "", f + "commit_aad9340c/water21.prm", hippo + "dispersionterm only\n", edisp},
   };
   for (const auto& c : cases) {
      TestFile fx1(c.xyz, xn);
      TestFile fk1(c.key, kn, c.extra);
      TestFile fp1(c.prm);
      TestSession session(argc, argv);
      session.init();
      REQUIRE(pltfm_config & Platform::ACC);
      REQUIRE_THROWS_WITH(c.term(calc::v7), Catch::Matchers::ContainsSubstring("do not take version"));
      session.end();
   }
}
#endif
