#include "ff/echglj.h"
#include "ff/nblist.h"
#include "test.h"
#include "testrt.h"

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
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

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
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

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
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

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
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

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
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
      }

      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_REALS(energy_elec, ref_ec, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_REALS(energy_elec, ref_ec, eps_e);

      session.end();
   }
}
