#include "ff/evalence.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

// Port of test_vasopressin in tinker/test/test_amoeba.f: the full AMOEBA
// potential of a 142-atom peptide in the gas phase with no cutoffs, so every
// valence and nonbonded term runs in one energy() call and the total is checked
// at each calc level rather than one term at a time.

// Measured against the reference, mixed precision is off by 2.1e-4 in the total
// energy, 1.1e-4 in a term energy, 8.4e-4 in the gradient and 1.5e-3 in the
// virial; double by 1.5e-6, 1.4e-6, 5.3e-6 and the 5e-4 rounding of the virial,
// which Tinker prints to only 3 decimals.
namespace {
const double eps_e = testGetEps(5.0e-4, 1.0e-5);
const double eps_g = testGetEps(1.3e-3, 2.0e-5);
const double eps_v = testGetEps(3.0e-3, 1.0e-3);
}

TEST_CASE("AMOEBA-Vasopressin-Alyz", "[ff][amoeba][vasopressin]")
{
   rc_flag = calc::xyz | calc::vmask;
   rc_flag |= calc::analyz;

   const char* xn = "test_vasopressin.xyz";
   const char* kn = "test_vasopressin.key";

   TestFile fx1(TINKER9_DIRSTR "/test/file/vasopressin/vasopressin.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/vasopressin/vasopressin.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6bd0b6fd/amoebabio18.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/vasopressin.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   TestSession session(argc, argv);
   session.init();
   REQUIRE(r.getGradientCount() == n);

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   COMPARE_VIR9(vir, ref_v, eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Bond Stretching", eng, cnt);
   COMPARE_INTS(nbond, cnt);
   COMPARE_ENERGY(eb, eng, eps_e);
   r.getEnergyCountByName("Angle Bending", eng, cnt);
   COMPARE_INTS(nangle, cnt);
   COMPARE_ENERGY(ea, eng, eps_e);
   r.getEnergyCountByName("Stretch-Bend", eng, cnt);
   COMPARE_INTS(nstrbnd, cnt);
   COMPARE_ENERGY(eba, eng, eps_e);
   r.getEnergyCountByName("Out-of-Plane Bend", eng, cnt);
   COMPARE_INTS(nopbend, cnt);
   COMPARE_ENERGY(eopb, eng, eps_e);
   r.getEnergyCountByName("Torsional Angle", eng, cnt);
   COMPARE_INTS(ntors, cnt);
   COMPARE_ENERGY(et, eng, eps_e);
   r.getEnergyCountByName("Pi-Orbital Torsion", eng, cnt);
   COMPARE_INTS(npitors, cnt);
   COMPARE_ENERGY(ept, eng, eps_e);
   r.getEnergyCountByName("Torsion-Torsion", eng, cnt);
   COMPARE_INTS(ntortor, cnt);
   COMPARE_ENERGY(ett, eng, eps_e);
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);

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

// Without calc::analyz the terms may share buffers or take the fused AMOEBA
// path, so only the totals are checked.
TEST_CASE("AMOEBA-Vasopressin", "[ff][amoeba][vasopressin]")
{
   rc_flag = calc::xyz | calc::vmask;

   const char* xn = "test_vasopressin.xyz";
   const char* kn = "test_vasopressin.key";

   TestFile fx1(TINKER9_DIRSTR "/test/file/vasopressin/vasopressin.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/vasopressin/vasopressin.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6bd0b6fd/amoebabio18.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/vasopressin.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const char* argv[] = {"dummy", xn};
   int argc = 2;

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
