#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "tool/platform.h"

#include "test.h"
#include "testrt.h"

#include <memory>
#include <string>
#include <vector>

using namespace tinker;

TEST_CASE("Truncated-Octahedron", "[ff][pbc][arbox]")
{
   rc_flag = calc::xyz | calc::vmask;

   const char* kname = "test_t8.key";
   const char* xname = "test_t8.xyz";

   const double eps_e = 0.0001;
   const double eps_g = 0.0002;
   const double eps_v = 0.001;
   const char* argv[] = {"dummy", xname};
   int argc = 2;

   TestFile fxy(TINKER9_DIRSTR "/test/file/trunc8/arbox.xyz", xname);
   TestFile fke(TINKER9_DIRSTR "/test/file/trunc8/arbox.key", kname);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/truncated8.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
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

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(countReduce(nev), ref_count);

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

// tinker9 has no rhombic dodecahedron box, so boxData() refuses DODECAHEDRON.
TEST_CASE("Dodecahedron-Unsupported", "[ff][pbc]")
{
   rc_flag = calc::xyz | calc::vmask;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const char* xyzname = "ions.xyz";
   const char* keyname = "dodecahedron.key";
   TestFile fxyz(files + "testvir/" + xyzname, xyzname);
   TestFile fkey(files + "testvir/01_charge_ewald.key", keyname, "dodecahedron\n");
   TestFile fprm(files + "commit_350df099/amber99sb.prm");

   const char* argv[] = {"dummy", xyzname, "-k", keyname};
   int argc = 4;
   TestSession session(argc, argv);
   REQUIRE_THROWS(session.init());
}

namespace {
struct OctTerm
{
   const char* name;
   CountBuffer& count;
   EnergyBuffer& energy;
};

// Runs the 777-atom water box in the octahedron of trunc8/octwater.xyz under
// \c key and compares the energy, gradient and virial of every version to the
// Fortran \c ref; with analysis, also the energy and count of each of \c terms.
// The 0.5 Angstrom list buffer in the keys lets the spatial list image a block
// of atoms once, the path that used to drop pairs across the hexagonal faces.
void runOctWater(const char* key, const char* prm, const char* ref, bool alyz, const std::vector<OctTerm>& terms)
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   if (alyz)
      rc_flag |= calc::analyz;

   const char* xn = "test_oct.xyz";
   const char* kn = "test_oct.key";
   std::string ke;
#if TINKER_GPULANG_OPENACC
   ke += "\ngpu-package openacc\n";
#endif

   const std::string files = TINKER9_DIRSTR "/test/file/";
   TestFile fx(files + "trunc8/octwater.xyz", xn);
   TestFile fk(files + "trunc8/" + key, kn, ke);
   std::unique_ptr<TestFile> fp;
   if (prm)
      fp.reset(new TestFile(files + prm));

   TestReference r(std::string(TINKER9_DIRSTR) + "/test/ref/" + ref);
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.03, 0.0001);
   const double eps_g = testGetEps(0.002, 0.0001);
   const double eps_v = testGetEps(0.05, 0.001);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   TestSession session(argc, argv);
   session.init();
#if TINKER_GPULANG_OPENACC
   REQUIRE(pltfm_config & Platform::ACC);
#endif
   if (use(Potent::CHARGE) and not alyz and (clistVersion() & Nbl::SPATIAL))
      REQUIRE(echgljAvailable());

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   if (alyz) {
      energy(calc::v3);
      COMPARE_REALS(esum, ref_e, eps_e);
      for (const auto& t : terms) {
         double eng;
         int cnt;
         r.getEnergyCountByName(t.name, eng, cnt);
         COMPARE_COUNT(t.count, cnt);
         COMPARE_ENERGY(t.energy, eng, eps_e);
      }
   }

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

void runOctAmoeba(bool alyz)
{
   runOctWater("octamoeba.key", "commit_6fe8e913/water03.prm", "truncated8.2.txt", alyz,
      {{"Van der Waals", nev, ev}, {"Atomic Multipoles", nem, em}, {"Polarization", nep, ep}});
}

void runOctHippo(bool alyz)
{
   runOctWater("octhippo.key", "commit_aad9340c/water21.prm", "truncated8.3.txt", alyz,
      {{"Repulsion", nrep, er}, {"Dispersion", ndisp, edsp}, {"Atomic Multipoles", nem, em},
         {"Polarization", nep, ep}, {"Charge Transfer", nct, ect}});
}

void runOctAplus(bool alyz)
{
   runOctWater("octaplus.key", "aplus2022/AMOEBAplus_Org.prm", "truncated8.4.txt", alyz,
      {{"Van der Waals", nev, ev}, {"Atomic Multipoles", nem, em}, {"Polarization", nep, ep},
         {"Charge Transfer", nct, ect}});
}

void runOctTip3p(bool alyz)
{
   runOctWater("octtip3p.key", nullptr, "truncated8.5.txt", alyz, {{"Van der Waals", nev, ev}, {"Charge-Charge", nec, ec}});
}
}

TEST_CASE("Octahedron-AMOEBA-Alyz", "[ff][pbc][octahedron]")
{
   runOctAmoeba(true);
}

TEST_CASE("Octahedron-AMOEBA", "[ff][pbc][octahedron]")
{
   runOctAmoeba(false);
}

TEST_CASE("Octahedron-HIPPO-Alyz", "[ff][pbc][octahedron]")
{
   runOctHippo(true);
}

TEST_CASE("Octahedron-HIPPO", "[ff][pbc][octahedron]")
{
   runOctHippo(false);
}

TEST_CASE("Octahedron-AMOEBAplus-Alyz", "[ff][pbc][octahedron]")
{
   runOctAplus(true);
}

TEST_CASE("Octahedron-AMOEBAplus", "[ff][pbc][octahedron]")
{
   runOctAplus(false);
}

TEST_CASE("Octahedron-TIP3P-Alyz", "[ff][pbc][octahedron]")
{
   runOctTip3p(true);
}

// Without analysis the charge and LJ terms run in the fused echglj kernel.
TEST_CASE("Octahedron-TIP3P", "[ff][pbc][octahedron]")
{
   runOctTip3p(false);
}
