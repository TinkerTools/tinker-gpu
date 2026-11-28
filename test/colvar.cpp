#include "ff/colvar.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/efep.h"
#include "ff/energy.h"
#include "ff/eost.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "tool/darray.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/mutant.hh>
#include <vector>

using namespace tinker;

namespace {
const char* kXyz = "test_colvar.xyz";
const char* kKey = "test_colvar.key";

// three waters; the second one sits a box edge (18.643) away from the first
const char* kWater3 = "     9  Water\n"
                      "     1  O     -2.037380    1.597451   -3.481484     1     2     3\n"
                      "     2  H     -1.125125    1.412994   -3.213200     2     1\n"
                      "     3  H     -2.541095    1.754841   -2.666805     2     1\n"
                      "     4  O     17.790469   -0.894103    0.301909     1     5     6\n"
                      "     5  H     17.534545   -1.104566   -0.607506     2     4\n"
                      "     6  H     18.112422    0.011819    0.268479     2     4\n"
                      "     7  O      1.259139   -2.415289    1.790163     1     8     9\n"
                      "     8  H      1.999646   -1.796414    1.703920     2     7\n"
                      "     9  H      0.543913   -2.030812    1.270042     2     7\n";

struct Ref
{
   double q;
   std::vector<int> atom;
   std::vector<double> grad;
   double w[9];
};

// the device coordinates, which the CVs see
std::vector<double> devicePos(const real* p)
{
   std::vector<double> h(n);
   darray::copyout(g::q0, n, h.data(), p);
   waitFor(g::q0);
   return h;
}

// the minimum image in a cubic box of edge a, or none for a = 0
void imageCubic(double r[3], double a)
{
   if (a > 0)
      for (int k = 0; k < 3; ++k)
         r[k] -= a * std::round(r[k] / a);
}

// the mass-weighted center of a set, which images the atoms in a molecule
// other than that of its first atom
void center(const std::vector<int>& set, const std::vector<double> (&p)[3], double a, double c[3])
{
   const int k0 = set[0];
   double m = 0;
   for (int k : set)
      m += atomid::mass[k];
   double d[3] = {0, 0, 0};
   for (int k : set) {
      double r[3] = {p[0][k] - p[0][k0], p[1][k] - p[1][k0], p[2][k] - p[2][k0]};
      if (molcul::molcule[k] != molcul::molcule[k0])
         imageCubic(r, a);
      for (int j = 0; j < 3; ++j)
         d[j] += atomid::mass[k] / m * r[j];
   }
   for (int j = 0; j < 3; ++j)
      c[j] = p[j][k0] + d[j];
}

// the reference DISTANCE CV between the centers of sets sa and sb
Ref refDistance(const std::vector<int>& sa, const std::vector<int>& sb, double a)
{
   const std::vector<double> p[3] = {devicePos(x), devicePos(y), devicePos(z)};
   double ca[3], cb[3];
   center(sa, p, a, ca);
   center(sb, p, a, cb);
   double r[3] = {ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]};
   if (molcul::molcule[sa[0]] != molcul::molcule[sb[0]])
      imageCubic(r, a);
   Ref f;
   f.q = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
   const double u[3] = {r[0] / f.q, r[1] / f.q, r[2] / f.q};
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         f.w[3 * i + j] = r[i] * u[j];
   double ma = 0, mb = 0;
   for (int k : sa)
      ma += atomid::mass[k];
   for (int k : sb)
      mb += atomid::mass[k];
   for (int k : sa) {
      f.atom.push_back(k);
      for (int j = 0; j < 3; ++j)
         f.grad.push_back(atomid::mass[k] / ma * u[j]);
   }
   for (int k : sb) {
      f.atom.push_back(k);
      for (int j = 0; j < 3; ++j)
         f.grad.push_back(-atomid::mass[k] / mb * u[j]);
   }
   return f;
}

void compareCv(const char* name, const Ref& f, double eps)
{
   INFO(name);
   const int i = colvarFind(name);
   REQUIRE(i >= 0);
   ColvarValue v;
   colvarCopyOut(i, v);
   REQUIRE(v.atom == f.atom);
   REQUIRE(v.grad.size() == f.grad.size());
   REQUIRE(v.q == Approx(f.q).margin(eps));
   for (size_t k = 0; k < f.grad.size(); ++k)
      REQUIRE(v.grad[k] == Approx(f.grad[k]).margin(eps));
   for (int k = 0; k < 9; ++k)
      REQUIRE(v.w[k] == Approx(f.w[k]).margin(eps));
}

// Only the CVs a LAMBDA-CV line couples are set up, so the keys couple every
// CV a case reads back.
const int kUsage = calc::xyz | calc::mass | calc::vmask;
}

TEST_CASE("Colvar-distance-pair", "[ff][colvar][L1]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   TestFile fke("", kKey,
      "parameters amoeba09\n"
      "cvterm only\n"
      "lambda 0.5\n"
      "CV-Atoms  Na  1\n"
      "cv-atoms  cl  2\n"
      "cv  r  Distance  na  CL\n"
      "lambda-cv  r  1.0  2.0  3.0\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   REQUIRE(colvarCount() == 1);
   REQUIRE(colvarFind("R") == 0);
   REQUIRE(colvarFind("r") == 0);
   REQUIRE(colvarFind("q") == -1);
   REQUIRE(colvarName(0) == "R");

   energy(calc::v0);
   ColvarValue v;
   colvarCopyOut(0, v);
   // Na at the origin, Cl at x = 2.2; dq/dx of Na points away from Cl
   const double eps = 1.0e-6;
   REQUIRE(v.q == Approx(2.2).margin(eps));
   REQUIRE(v.atom == std::vector<int>{0, 1});
   const double g[] = {-1, 0, 0, 1, 0, 0};
   for (int k = 0; k < 6; ++k)
      REQUIRE(v.grad[k] == Approx(g[k]).margin(eps));
   for (int k = 0; k < 9; ++k)
      REQUIRE(v.w[k] == Approx(k == 0 ? 2.2 : 0.0).margin(eps));
   compareCv("r", refDistance({0}, {1}, 0), eps);

   session.end();
}

TEST_CASE("Colvar-distance-image", "[ff][colvar][L1]")
{
   // Na and Cl are two molecules on the opposite faces of the box
   const double a = 18.643;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy("", kXyz,
      "     2  NaCl\n"
      "     1  Na+   -9.000000    0.200000    0.100000     7\n"
      "     2  Cl-    8.900000   -0.300000    0.500000    15\n");
   TestFile fke("", kKey,
      "parameters amoeba09\n"
      "cvterm only\n"
      "lambda 0.5\n"
      "a-axis 18.643\n"
      "cv-atoms  na  1\n"
      "cv-atoms  cl  2\n"
      "cv  r  distance  na  cl\n"
      "lambda-cv  r  1.0  0.0  1.0\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   energy(calc::v0);
   ColvarValue v;
   colvarCopyOut(0, v);
   const double dx = -9.0 - 8.9 + a, dy = 0.5, dz = -0.4;
   const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
   const double eps = 1.0e-5;
   REQUIRE(v.q == Approx(r).margin(eps));
   REQUIRE(v.grad[0] == Approx(dx / r).margin(eps));
   REQUIRE(v.grad[3] == Approx(-dx / r).margin(eps));
   compareCv("r", refDistance({0}, {1}, a), eps);

   session.end();
}

TEST_CASE("Colvar-distance-com", "[ff][colvar][L1]")
{
   // Three waters; the second one sits a box edge away from the first, so the
   // set made of the first oxygen and the second water must image its atoms.
   const double a = 18.643;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey,
      "parameters water03\n"
      "cvterm only\n"
      "lambda 0.5\n"
      "a-axis 18.643\n"
      "cv-atoms  pair  1\n"
      "cv-atoms  pair  4,-5 6\n"
      "cv-atoms  w3    -7 9\n"
      "cv-atoms  h     2 3\n"
      "cv-atoms  odd   1 4 5 8\n"
      "cv  d1  distance  pair  w3\n"
      "cv  d2  distance  h     w3\n"
      "cv  d3  distance  odd   h\n"
      "lambda-cv  d1  1.0  0.0  1.0\n"
      "lambda-cv  d2  1.0  0.0  1.0\n"
      "lambda-cv  d3  1.0  0.0  1.0\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   REQUIRE(colvarCount() == 3);
   energy(calc::v0);
   const double eps = 1.0e-5;
   const Ref d1 = refDistance({0, 3, 4, 5}, {6, 7, 8}, a);
   compareCv("d1", d1, eps);
   compareCv("d2", refDistance({1, 2}, {6, 7, 8}, a), eps);
   // a set of scattered atoms from all three waters
   compareCv("d3", refDistance({0, 3, 4, 7}, {1, 2}, a), eps);

   // without imaging the second water, the center of "pair" would be off by
   // most of a box edge
   REQUIRE(d1.q < 5.0);

   // dq/dx sums to zero over the slots
   ColvarValue v;
   colvarCopyOut(colvarFind("d1"), v);
   for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (size_t k = 0; k < v.atom.size(); ++k)
         s += v.grad[3 * k + j];
      REQUIRE(s == Approx(0).margin(eps));
   }

   session.end();
}

TEST_CASE("Colvar-keys", "[ff][colvar][L1]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   const char* argv[] = {"dummy", kXyz};

   // the CV lines are read only with the restraint on: CVTERM and a LAMBDA-CV
   // line; the sets and CVs are parsed before the couplings, wherever the
   // lines are, so their errors come first
   const std::string on = "parameters amoeba09\ncvterm only\nlambda 0.5\nlambda-cv r 1.0 0.0 1.0\n";
   auto refused = [&](const std::string& keys, const char* message) {
      INFO(keys);
      TestFile fke("", kKey, on + keys);
      TestSession session(2, argv);
      rc_flag = kUsage;
      REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring(message));
   };
   const std::string sets = "cv-atoms na 1\ncv-atoms cl 2\n";
   refused(sets + "cv r distance na\n", "Takes Two Atom Sets");
   refused(sets + "cv r distance na cl na\n", "Takes Two Atom Sets");
   refused(sets + "cv r distance na xx\n", "Set XX of CV R Is Not Defined by CV-ATOMS");
   refused(sets + "cv r distance na na\n", "Needs Two Different Atom Sets");
   refused(sets + "cv r distance na cl\ncv R distance cl na\n", "CV R Is Defined Twice");
   refused(sets + "cv r angle na cl\n", "Unknown Type ANGLE of CV R");
   refused(sets + "cv 1r distance na cl\n", "Must Start with a Letter");
   refused(sets + "cv r\n", "Give the Name and the Type");
   refused("cv-atoms na\n", "Give the Name and the Atoms");
   refused("cv-atoms 2na 1\n", "Must Start with a Letter");
   refused("cv-atoms na 1\ncv-atoms na 1\n", "Atom 1 Is Listed Twice in Set NA");
   refused("cv-atoms na 3\n", "Atom 3 of Set NA Is Out of Range");
   refused("cv-atoms na 0\n", "Atom 0 of Set NA Is Out of Range");
   refused("cv-atoms na 1-2\n", "Cannot Read the Atoms of Set NA");
   refused("cv-atoms na x\n", "Cannot Read the Atoms of Set NA");
   refused("cv-atoms na -2 1\n", "Atom Range 2 to 1 of Set NA Is Empty");

   // the centers of the sets need the atomic masses on the device
   {
      TestFile fke("", kKey, on + sets + "cv r distance na cl\n");
      TestSession session(2, argv);
      rc_flag = calc::xyz | calc::vmask;
      REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring("Atomic Masses the CVs Need"));
   }

   // with the restraint off nothing is set up
   TestFile fke("", kKey, "parameters amoeba09\nvdwterm only\n" + sets + "cv r distance na cl\n");
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();
   REQUIRE_FALSE(use(Potent::COLVAR));
   REQUIRE(colvarCount() == 0);
   session.end();
}

TEST_CASE("Colvar-uncoupled", "[ff][colvar][L1]")
{
   // A CV no LAMBDA-CV line couples is read and checked, then left out with
   // the atom sets only it uses; the coupled CV keeps its own sets.
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/water10/h2o10.xyz", kXyz);
   TestFile fke("", kKey,
      "parameters water03\n"
      "cvterm only\n"
      "lambda 0.5\n"
      "cv-atoms  lig   10 11 12\n"
      "cv-atoms  pock  3 7\n"
      "cv-atoms  na    20\n"
      "cv  d1  distance  lig  pock\n"
      "cv  d2  distance  na   lig\n"
      "lambda-cv  d2  5.0  3.0  6.0\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   REQUIRE(colvarCount() == 1);
   REQUIRE(colvarFind("d1") == -1);
   REQUIRE(colvarFind("d2") == 0);
   REQUIRE(countBondedTerm(Potent::COLVAR) == 1);
   const auto& c = colvar;
   REQUIRE(c.nset == 2); // na and lig; pock is left out with d1
   REQUIRE(c.nsat == 4);
   REQUIRE(c.nslot == 4);
   energy(calc::v0);
   compareCv("d2", refDistance({19}, {9, 10, 11}, 0), 1.0e-5);
   session.end();
}

TEST_CASE("Colvar-distance-large", "[ff][colvar][L1]")
{
   // A 2700-atom water box. Set A is 100 whole waters, so whole warps hold
   // one set and add their sums at once; set B is 800 scattered oxygens given
   // on 80 lines that append; a warp straddles the two. CV "o" shares set B.
   const double a = 30.0;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", kXyz);
   std::string key = "parameters water03\ncvterm only\nlambda 0.5\ncv-atoms a -1 300\ncv-atoms o 2700\n";
   std::vector<int> sb;
   for (int line = 0; line < 80; ++line) {
      key += "cv-atoms b";
      for (int i = 0; i < 10; ++i) {
         const int k = 300 + 3 * (10 * line + i); // 0-based oxygen of waters 101 to 900
         key += " " + std::to_string(k + 1);
         sb.push_back(k);
      }
      key += "\n";
   }
   key += "cv big distance a b\ncv o distance b o\nlambda-cv big 1.0 0.0 1.0\nlambda-cv o 1.0 0.0 1.0\n";
   TestFile fke("", kKey, key);
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   std::vector<int> sa;
   for (int k = 0; k < 300; ++k)
      sa.push_back(k);
   energy(calc::v0);
   const double eps = 1.0e-4;
   compareCv("big", refDistance(sa, sb, a), eps);
   compareCv("o", refDistance(sb, {2699}, a), eps);

   // a second evaluation starts the sums from zero again
   energy(calc::v0);
   compareCv("big", refDistance(sa, sb, a), eps);

   session.end();
}

TEST_CASE("Colvar-layout", "[ff][colvar][L1]")
{
   // the example in the colvar page of include/ff/colvar.h, array by array
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/water10/h2o10.xyz", kXyz);
   TestFile fke("", kKey,
      "parameters water03\n"
      "cvterm only\n"
      "lambda 0.5\n"
      "cv-atoms  lig   10 11 12\n"
      "cv-atoms  pock  3 7\n"
      "cv-atoms  na    20\n"
      "cv  d1  distance  lig  pock\n"
      "cv  d2  distance  na   lig\n"
      "lambda-cv  d1  10.0  2.5  8.0\n"
      "lambda-cv  d2   5.0  3.0  6.0\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   const auto& c = colvar;
   REQUIRE(c.nset == 3);
   REQUIRE(c.nsat == 6);
   REQUIRE(c.ncv == 2);
   REQUIRE(c.nslot == 9);

   auto ints = [](int nelem, const int* p) {
      std::vector<int> h(nelem);
      darray::copyout(g::q0, nelem, h.data(), p);
      waitFor(g::q0);
      return h;
   };
   REQUIRE(ints(6, c.satom) == std::vector<int>{9, 10, 11, 2, 6, 19});
   REQUIRE(ints(6, &c.srange[0][0]) == std::vector<int>{0, 3, 3, 5, 5, 6});
   REQUIRE(ints(6, c.sown) == std::vector<int>{0, 0, 0, 1, 1, 2});
   REQUIRE(ints(2, c.type) == std::vector<int>{(int)ColvarType::DISTANCE, (int)ColvarType::DISTANCE});
   const auto iarg = ints(8, &c.iarg[0][0]);
   REQUIRE(iarg[0] == 0);
   REQUIRE(iarg[1] == 1);
   REQUIRE(iarg[4] == 2);
   REQUIRE(iarg[5] == 0);
   REQUIRE(ints(4, &c.slot[0][0]) == std::vector<int>{0, 5, 5, 9});
   REQUIRE(ints(9, c.gatom) == std::vector<int>{9, 10, 11, 2, 6, 19, 9, 10, 11});
   REQUIRE(ints(9, c.gcv) == std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1});

   energy(calc::v0);
   const double eps = 1.0e-5;
   compareCv("d1", refDistance({9, 10, 11}, {2, 6}, 0), eps);
   const Ref d2 = refDistance({19}, {9, 10, 11}, 0);
   compareCv("d2", d2, eps);
   // slot 7, the third slot of d2, is stored atom 10 (input atom 11) of lig,
   // on the B side:
   // grad = -(m10/M_lig) u, where u is the gradient of na, the one atom of A
   ColvarValue v;
   colvarCopyOut(1, v);
   REQUIRE(v.atom[2] == 10);
   const double mlig = atomid::mass[9] + atomid::mass[10] + atomid::mass[11];
   for (int k = 0; k < 3; ++k)
      REQUIRE(v.grad[6 + k] == Approx(-atomid::mass[10] / mlig * v.grad[k]).margin(eps));
   session.end();
}

//====================================================================//
// LAMBDA-CV

namespace {
// the LAMBDA-CV restraint E = K (q - q0)^2 with q0 = qmin + L (qmax - qmin)
struct Couple
{
   double k, qmin, qmax;
};

struct CoupleExpect
{
   double e;      // energy
   double de;     // dE/dq
   double dedl;   // dE/dL
   double d2edl2; // d2E/dL2
   double fdl;    // d(dE/dL)/dq
};

// with the collective variable sub-lambda g and its derivatives g1, g2
CoupleExpect expectCoupleMap(const Couple& c, double q, double g, double g1, double g2)
{
   const double dq = c.qmax - c.qmin;
   const double dev = q - (c.qmin + g * dq);
   const double dq0 = g1 * dq;
   return {c.k * dev * dev, 2 * c.k * dev, -2 * c.k * dq0 * dev, 2 * c.k * dq0 * dq0 - 2 * c.k * dev * g2 * dq,
      -2 * c.k * dq0};
}

// the sub-lambda following the main lambda one to one
CoupleExpect expectCouple(const Couple& c, double q, double lmda)
{
   return expectCoupleMap(c, q, lmda, 1, 0);
}

const Couple kNacl = {10.0, 2.0, 4.0};

// a lambda derivative needs the map of the sub-lambda it drives
// (mutate.f:mutate_dlmda); this one follows the main lambda one to one
const std::string kMap = "cv-lmda-map exp\n";
const Couple kW3 = {5.0, 1.0, 6.0};
const int kRcPlain = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;

// Na at the origin and Cl at x = 2.2, the restraint alone
std::string naclKey(const std::string& extra)
{
   return "parameters amoeba09\ncvterm only\ncv-atoms na 1\ncv-atoms cl 2\ncv r distance na cl\n" + extra;
}

// the three waters of kWater3 in a box, a CV across two molecules and the image
std::string water3Key(const std::string& extra)
{
   return "parameters water03\ncvterm only\na-axis 18.643\n"
          "cv-atoms pair 1\ncv-atoms pair 4,-5 6\ncv-atoms w3 -7 9\ncv d1 distance pair w3\n"
      + extra;
}

// f times the slot gradients of CV i, summed per atom into a 3n array
std::vector<double> slotGradient(int i, double f)
{
   ColvarValue v;
   colvarCopyOut(i, v);
   std::vector<double> g(3 * n, 0.0);
   for (size_t s = 0; s < v.atom.size(); ++s)
      for (int j = 0; j < 3; ++j)
         g[3 * v.atom[s] + j] += f * v.grad[3 * s + j];
   return g;
}

ColvarValue evalCv(int i)
{
   energy(calc::v0);
   ColvarValue v;
   colvarCopyOut(i, v);
   return v;
}

}

TEST_CASE("Colvar-lambda-keys", "[ff][colvar][L1]")
{
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   const char* argv[] = {"dummy", kXyz};

   auto refused = [&](const std::string& extra, const char* message) {
      INFO(extra);
      TestFile fke("", kKey, naclKey(extra));
      TestSession session(2, argv);
      rc_flag = kUsage;
      REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring(message));
   };
   refused("lambda-cv r 10.0 2.0 4.0\n", "Coupling a CV to Lambda Requires the LAMBDA Keyword");
   refused("lambda 0.5\nlambda-cv q 10.0 2.0 4.0\n", "CV Q Is Not Defined by CV");
   refused("lambda 0.5\nlambda-cv r 10.0 2.0\n", "Give the CV, the Force Constant, and qmin and qmax");
   refused("lambda 0.5\nlambda-cv r ten 2.0 4.0\n", "Cannot Read the Force Constant and Range of CV R");
   refused("lambda 0.5\nlambda-cv r 10.0 2.0 4.0x\n", "Cannot Read the Force Constant and Range of CV R");
   refused("lambda 0.5\nlambda-cv r 0.0 2.0 4.0\n", "The Force Constant of CV R Must Be Positive");
   refused("lambda 0.5\nlambda-cv r 10.0 2.0 2.0\n", "CV R Needs Different qmin and qmax");
   refused("lambda 0.5\nlambda-cv r 10.0 2.0 4.0\nlambda-cv R 5.0 2.0 4.0\n", "CV R Is Coupled Twice");
   refused("lambda 0.5\nlambda-cv r 10.0 nan 4.0\n", "Range of CV R Must Be Finite");
   refused("lambda 0.5\nlambda-cv r 10.0 2.0 inf\n", "Range of CV R Must Be Finite");
   refused("lambda 0.5\nligand 2\nlambda-cv r 10.0 2.0 4.0\n", "Mutated Atoms and LAMBDA-CV Cannot Share");
}

TEST_CASE("Colvar-lambda-term", "[ff][colvar][L1]")
{
   // CVTERM and the ONLY keywords of the other terms switch the restraint
   // like any term (potent::use_colvar): CVTERM NONE leaves no restraint,
   // and VDWTERM ONLY gives the same energy as the key without LAMBDA-CV
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   const char* argv[] = {"dummy", kXyz};
   const std::string cvs = "cv-atoms na 1\ncv-atoms cl 2\ncv r distance na cl\nlambda 0.3\n";
   const std::string couple = "lambda-cv r 10.0 2.0 4.0\n";

   auto run = [&](const std::string& key, bool on) {
      INFO(key);
      TestFile fke("", kKey, "parameters amoeba09\n" + key);
      TestSession session(2, argv);
      rc_flag = kUsage;
      session.init();
      REQUIRE(use(Potent::COLVAR) == on);
      REQUIRE(countBondedTerm(Potent::COLVAR) == (on ? 1 : 0));
      energy(calc::v3);
      const double e = esum;
      session.end();
      return e;
   };
   const double eps = 1.0e-6;
   REQUIRE(run("cvterm only\n" + cvs + couple, true) == Approx(expectCouple(kNacl, 2.2, 0.3).e).margin(eps));
   REQUIRE(run("restrainterm only\ncvterm none\n" + cvs + couple, false) == Approx(0.0).margin(eps));
   const double evdw = run("vdwterm only\n" + cvs, false);
   REQUIRE(run("vdwterm only\n" + cvs + couple, false) == Approx(evdw).margin(eps));
   REQUIRE(evdw != 0);
}

TEST_CASE("Colvar-lambda-mcstate", "[ff][colvar][L1]")
{
   // a lambda Monte Carlo trial that is turned down puts back the lambda
   // derivatives of the step, those of the LAMBDA-CV channel too
   decvdl = 1.5;
   d2ecvdl2 = 2.5;
   const LmdaStepState step;
   decvdl = -7.0;
   d2ecvdl2 = -9.0;
   step.restore();
   REQUIRE(decvdl == 1.5);
   REQUIRE(d2ecvdl2 == 2.5);
   decvdl = 0;
   d2ecvdl2 = 0;
}

// Lambda derivatives and lambda dynamics need the CUDA platform (dlmdaData2).
#if TINKER_GPULANG_CUDA
TEST_CASE("Colvar-lambda-analytic", "[ff][colvar][L1]")
{
   // Under analysis the term keeps its energy and its lambda derivatives in
   // its own buffers and channel; the restraint is the only term.
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   TestFile fke("", kKey, naclKey("lambda 0.3\nlambda-deriv2\nlambda-cv r 10.0 2.0 4.0\n" + kMap));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();
   REQUIRE(use(Potent::COLVAR));
   REQUIRE(countBondedTerm(Potent::COLVAR) == 1);
   REQUIRE(use_cvdlmda);

   const double eps = 1.0e-4;
   // the lambda of the key, then moved as a lambda move would
   for (double l : {0.3, 0.0, 1.0}) {
      INFO(l);
      lambda = l;
      energy(calc::v1);
      const auto x = expectCouple(kNacl, 2.2, l);
      COMPARE_REALS(esum, x.e, eps);
      COMPARE_REALS(energy_ecv, x.e, eps);
      COMPARE_REALS(dedl, x.dedl, eps);
      COMPARE_REALS(decvdl, x.dedl, eps);
      COMPARE_REALS(d2edl2, x.d2edl2, eps);
      COMPARE_REALS(d2ecvdl2, x.d2edl2, eps);

      // dq/dx is (-1,0,0) on Na and (1,0,0) on Cl; W = diag(2.2, 0, 0)
      std::vector<double> g, f;
      copyGradientFlat(calc::grad, g);
      copyGradientFlat(calc::grad, f, dfdlx, dfdly, dfdlz);
      testRequireNear(g, {-x.de, 0, 0, x.de, 0, 0}, eps);
      testRequireNear(f, {-x.fdl, 0, 0, x.fdl, 0, 0}, eps);
      for (int k = 0; k < 9; ++k) {
         INFO(k);
         REQUIRE(vir[k] == Approx(k == 0 ? x.de * 2.2 : 0.0).margin(eps));
         REQUIRE(dvirdl[k] == Approx(k == 0 ? x.fdl * 2.2 : 0.0).margin(eps));
      }
   }
   lambda = 0.3;
   session.end();
}

TEST_CASE("Colvar-lambda-fd", "[ff][colvar][L2]")
{
   // testlmda, analytical against numerical, for a CV between two molecules
   // that needs the image; the restraint is quadratic in lambda, so a large
   // step costs no truncation error
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey, water3Key("lambda 0.37\nlambda-deriv2\nlambda-cv d1 5.0 1.0 6.0\n" + kMap));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   FdTestOptions opts;
   opts.eps = 0.05;
   rc_flag = testlmdaFlags(opts);
   session.init();

   const auto r = testlmdaEvaluate(opts);
   const auto v = evalCv(0);
   const auto x = expectCouple(kW3, v.q, 0.37);
   COMPARE_REALS(r.dedl[0], x.dedl, 1.0e-3);
   COMPARE_REALS(r.ndedl[0], x.dedl, 1.0e-3);
   COMPARE_REALS(r.d2edl2[0], x.d2edl2, 1.0e-3);
   COMPARE_REALS(r.nd2edl2[0], x.d2edl2, 1.0e-2);
   // nothing in the van der Waals, multipole or polarization columns
   for (int k = 1; k < nlmdacol; ++k) {
      REQUIRE(r.dedl[k] == 0);
      REQUIRE(r.d2edl2[k] == 0);
   }
   testRequireNear(r.dfdl, slotGradient(0, x.fdl), 1.0e-4);
   testRequireNear(r.ndfdl, r.dfdl, 1.0e-3);
   for (int k = 0; k < 9; ++k) {
      INFO(k);
      REQUIRE(r.dvirdl[k] == Approx(x.fdl * v.w[k]).margin(1.0e-3));
      REQUIRE(r.ndvirdl[k] == Approx(r.dvirdl[k]).margin(1.0e-3));
   }
   session.end();
}

#endif

TEST_CASE("Colvar-lambda-gradvir", "[ff][colvar][L2]")
{
   // testgrad and testvir with the restraint at the lambda of the key, which
   // applies without LAMBDA-DERIV
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey, water3Key("lambda 0.37\nlambda-cv d1 5.0 1.0 6.0\n"));
   const char* argv[] = {"dummy", kXyz};

   {
      TestSession session(2, argv);
      FdTestOptions opts;
      opts.eps = 1.0e-2;
      rc_flag = testgradFlags(opts);
      session.init();
      REQUIRE_FALSE(use_cvdlmda);
      const auto r = testgradEvaluate(opts);
      const auto v = evalCv(0);
      const auto x = expectCouple(kW3, v.q, 0.37);
      REQUIRE(r.energy == Approx(x.e).margin(1.0e-4));
      testRequireNear(r.ganlyt, slotGradient(0, x.de), 1.0e-4);
      testRequireNear(r.gnumer, r.ganlyt, 1.0e-2);
      session.end();
   }
   {
      TestSession session(2, argv);
      rc_flag = testvirFlags();
      session.init();
      const auto r = testvirEvaluate(0.02);
      REQUIRE(r.numer);
      const auto v = evalCv(0);
      const auto x = expectCouple(kW3, v.q, 0.37);
      for (int k = 0; k < 9; ++k) {
         INFO(k);
         REQUIRE(r.vanlyt[k] == Approx(x.de * v.w[k]).margin(1.0e-3));
         REQUIRE(r.vnumer[k] == Approx(r.vanlyt[k]).margin(2.0e-2));
      }
      session.end();
   }
}

// Lambda derivatives and lambda dynamics need the CUDA platform (dlmdaData2).
#if TINKER_GPULANG_CUDA
TEST_CASE("Colvar-lambda-versions", "[ff][colvar][L2]")
{
   // Without analysis the term adds into the shared buffers; every version
   // agrees, and only v1 and v4 take the lambda derivatives (v9 and v10 with
   // LAMBDA-DERIV2, v7 and v8 with LAMBDA-DERIV, none without).
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   const char* argv[] = {"dummy", kXyz};

   auto run = [&](const std::string& deriv) {
      INFO(deriv);
      TestFile fke("", kKey, water3Key("lambda 0.37\n" + deriv + "lambda-cv d1 5.0 1.0 6.0\n" + kMap));
      TestSession session(2, argv);
      rc_flag = kRcPlain;
      session.init();
      testRequireSharedLmdaBuffers();

      const auto v = evalCv(0);
      const auto x = expectCouple(kW3, v.q, 0.37);
      const auto ge = slotGradient(0, x.de);
      const auto fe = slotGradient(0, x.fdl);
      const double eps = 1.0e-4;
      for (int vers : {calc::v0, calc::v1, calc::v4, calc::v5, calc::v6}) {
         INFO(vers);
         energy(vers);
         if (vers & calc::energy)
            COMPARE_REALS(esum, x.e, eps);
         if (vers & calc::grad) {
            std::vector<double> g;
            copyGradientFlat(calc::grad, g);
            testRequireNear(g, ge, eps);
         }
         if (vers & calc::virial)
            for (int k = 0; k < 9; ++k)
               REQUIRE(vir[k] == Approx(x.de * v.w[k]).margin(eps));
         if (vers == calc::v1 or vers == calc::v4) {
            COMPARE_REALS(dedl, use_dlmda ? x.dedl : 0.0, eps);
            COMPARE_REALS(d2edl2, use_d2lmda ? x.d2edl2 : 0.0, eps);
            if (use_d2lmda) {
               std::vector<double> f;
               copyGradientFlat(calc::grad, f, dfdlx, dfdly, dfdlz);
               testRequireNear(f, fe, eps);
               if (vers == calc::v1)
                  for (int k = 0; k < 9; ++k)
                     REQUIRE(dvirdl[k] == Approx(x.fdl * v.w[k]).margin(eps));
            }
         }
      }
      session.end();
   };
   run("lambda-deriv2\n");
   run("lambda-deriv\n");
   run("");
}

TEST_CASE("Colvar-lambda-ost", "[ff][colvar][L2]")
{
   // With no mutated atoms, OST sees the dE/dL of the restraint, and an energy
   // at another lambda, as a Monte Carlo trial or a BAR window evaluates it,
   // follows the lambda of the moment.
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   TestFile fke("", kKey, naclKey("lambda 0.5\nlambda-mode ost\nlambda-cv r 10.0 2.0 4.0\n" + kMap));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kRcPlain;
   session.init();
   REQUIRE(use_ost);
   REQUIRE(use_d2lmda); // OST takes the second lambda derivatives

   const double eps = 1.0e-4;
   energy(calc::v1);
   const auto x = expectCouple(kNacl, 2.2, 0.5);
   COMPARE_REALS(dedl, x.dedl, eps);
   COMPARE_REALS(d2edl2, x.d2edl2, eps);

   lambda = 0.8;
   energy(calc::v0);
   COMPARE_REALS(esum, expectCouple(kNacl, 2.2, 0.8).e, eps);
   lambda = 0.5;
   session.end();
}

TEST_CASE("Colvar-lambda-dynamic", "[ff][colvar][L3]")
{
   // 200 steps of stochastic dynamics of NaCl, driven by the restraint alone,
   // under each lambda method: OST, metadynamics and ABF, then TI and FEP over
   // two windows. Stochastic dynamics removes no motion of the center or
   // rotation, so the pair keeps its six degrees of freedom. Each run ends
   // cleanly (a Fortran stop exits 0, so read the log), with a frame at each
   // 0.1 ps and the file of its method; the TI blocks are at the windows.
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", "cvdyn.xyz");
   TestRemoveFileOnExit farc("cvdyn.arc"), fdyn("cvdyn.dyn"), flog("cvdyn.log");
   const std::string windows = "lambda-window 0.4\nlambda-window 0.6\nlambda-eqratio 0.5\n";
   const std::pair<const char*, std::string> modes[] = {{"ost", ""}, {"meta", ""}, {"abf", ""},
      {"ti", windows + "ti-nstepavg 10\n"}, {"fep", windows}};
   for (const auto& m : modes) {
      const std::string mode = m.first;
      INFO(mode);
      const std::string hist = "cvdyn." + mode;
      std::remove(hist.c_str());
      TestRemoveFileOnExit fhist(hist);
      TestFile fke("", "cvdyn.key",
         naclKey("lambda 0.5\nlambda-mode " + mode
            + "\nlambda-interval 10\nlambda-cv r 10.0 2.0 4.0\n"
              "integrator stochastic\nrandomseed 1\n"
            + m.second + kMap));
      const std::string cmd = std::string(TINKER9_BINDIRSTR)
         + "/tinker9 dynamic cvdyn.xyz 200 1.0 0.1 2 298 > cvdyn.log 2>&1";
      const int rc = std::system(cmd.c_str());
      std::ifstream log("cvdyn.log");
      const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
      INFO(text);
      REQUIRE(rc == 0);
      REQUIRE(text.find("Unable to Continue") == std::string::npos);
      std::ifstream farcin("cvdyn.arc");
      int lines = 0;
      for (std::string line; std::getline(farcin, line);)
         ++lines;
      REQUIRE(lines == 2 * 3); // two frames of a header and two atoms
      std::ifstream fh(hist);
      REQUIRE(fh.good());
      if (mode == "ti") {
         // the rows after the column labels: index, lambda, dE/dL, deviation
         int nblock = 0;
         bool rows = false;
         for (std::string line; std::getline(fh, line);) {
            if (line.find("Index") != std::string::npos) {
               rows = true;
               continue;
            }
            std::istringstream row(line);
            int i;
            double l, d, sd;
            if (rows and (row >> i >> l >> d >> sd)) {
               INFO(line);
               REQUIRE(i == ++nblock);
               REQUIRE((l == 0.4 or l == 0.6));
               REQUIRE(std::isfinite(d));
               REQUIRE(d != 0);
            }
         }
         REQUIRE(nblock == 10); // five blocks of ten steps in each window
      }
      std::remove("cvdyn.arc");
      std::remove("cvdyn.dyn");
   }
}
TEST_CASE("Colvar-lambda-qnt", "[ff][colvar][L2]")
{
   // A QNT window on the collective variable sub-lambda: the chain rule
   // through g' and g'', against the smoothstep g = u^3 (10 - 15u + 6u^2)
   // with u = (L - 0.2)/0.6, and testlmda analytical against numerical.
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey,
      water3Key("lambda 0.37\nlambda-deriv2\ncv-lmda-map qnt\ncv-lmda-range 0.2 0.8\nlambda-cv d1 5.0 1.0 6.0\n"));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   // unlike a linear map, the taper puts a large third derivative into E(L),
   // so the central differences need a small step
   FdTestOptions opts;
   opts.eps = 1.0e-3;
   rc_flag = testlmdaFlags(opts);
   session.init();

   const double w = 0.6, u = (0.37 - 0.2) / w;
   const double g = u * u * u * (10 - 15 * u + 6 * u * u);
   const double g1 = 30 * u * u * (1 - u) * (1 - u) / w;
   const double g2 = 60 * u * (1 - u) * (1 - 2 * u) / (w * w);
   REQUIRE(cvlam == Approx(g).epsilon(1.0e-12));
   REQUIRE(dcvldlmda == Approx(g1).epsilon(1.0e-12));
   REQUIRE(d2cvldlmda2 == Approx(g2).epsilon(1.0e-12));

   const auto r = testlmdaEvaluate(opts);
   const auto v = evalCv(0);
   const auto x = expectCoupleMap(kW3, v.q, g, g1, g2);
   REQUIRE(r.dedl[0] == Approx(x.dedl).epsilon(1.0e-4));
   REQUIRE(r.d2edl2[0] == Approx(x.d2edl2).epsilon(1.0e-4));
   REQUIRE(r.ndedl[0] == Approx(x.dedl).epsilon(1.0e-3));
   REQUIRE(r.nd2edl2[0] == Approx(x.d2edl2).epsilon(1.0e-2));
   testRequireNear(r.dfdl, slotGradient(0, x.fdl), 1.0e-3);
   testRequireNear(r.ndfdl, r.dfdl, 5.0e-2);
   for (int k = 0; k < 9; ++k) {
      INFO(k);
      REQUIRE(r.dvirdl[k] == Approx(x.fdl * v.w[k]).margin(1.0e-3));
      REQUIRE(r.ndvirdl[k] == Approx(r.dvirdl[k]).margin(5.0e-2));
   }
   session.end();
}

TEST_CASE("Colvar-lambda-held", "[ff][colvar][L2]")
{
   // Naming another map but not CV-LMDA-MAP holds the collective variable
   // sub-lambda at CV-LAMBDA: a fixed restraint that does not follow the main
   // lambda and adds nothing to its derivatives.
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey,
      water3Key("lambda 0.37\nlambda-deriv2\nvdw-lmda-map exp\ncv-lambda 0.6\nlambda-cv d1 5.0 1.0 6.0\n"));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kRcPlain;
   session.init();
   REQUIRE(cvlam == 0.6);
   REQUIRE(dcvldlmda == 0);
   REQUIRE_FALSE(use_cvdlmda); // no CV-LMDA-MAP (mutate.f:setdlmdaterms)
   REQUIRE_FALSE(cvdlmdaActive());

   const auto v = evalCv(0);
   const auto x = expectCoupleMap(kW3, v.q, 0.6, 0, 0);
   const double eps = 1.0e-4;
   for (double l : {0.37, 0.9}) {
      INFO(l);
      lambda = l;
      energy(calc::v1);
      COMPARE_REALS(esum, x.e, eps);
      COMPARE_REALS(dedl, 0.0, eps);
      COMPARE_REALS(d2edl2, 0.0, eps);
      std::vector<double> gr;
      copyGradientFlat(calc::grad, gr);
      testRequireNear(gr, slotGradient(0, x.de), eps);
   }
   lambda = 0.37;
   session.end();
}
TEST_CASE("Colvar-lambda-multi", "[ff][colvar][L2]")
{
   // two CVs coupled with their own K and range, both following the same
   // sub-lambda: the energy, gradient and lambda derivatives add up
   TestLmdaFlagReset lmdaReset;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   TestFile fke("", kKey,
      water3Key("lambda 0.37\nlambda-deriv2\ncv-atoms h 2 3\ncv d2 distance h w3\n"
                "lambda-cv d1 5.0 1.0 6.0\nlambda-cv d2 3.0 4.0 2.0\n"
         + kMap));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kRcPlain;
   session.init();
   REQUIRE(countBondedTerm(Potent::COLVAR) == 2);

   energy(calc::v1);
   ColvarValue v1, v2;
   colvarCopyOut(colvarFind("d1"), v1);
   colvarCopyOut(colvarFind("d2"), v2);
   const auto x1 = expectCouple(kW3, v1.q, 0.37);
   const auto x2 = expectCouple(Couple{3.0, 4.0, 2.0}, v2.q, 0.37);
   const double eps = 1.0e-4;
   COMPARE_REALS(esum, x1.e + x2.e, eps);
   COMPARE_REALS(dedl, x1.dedl + x2.dedl, eps);
   COMPARE_REALS(d2edl2, x1.d2edl2 + x2.d2edl2, eps);
   auto ge = slotGradient(colvarFind("d1"), x1.de);
   const auto ge2 = slotGradient(colvarFind("d2"), x2.de);
   for (size_t k = 0; k < ge.size(); ++k)
      ge[k] += ge2[k];
   std::vector<double> g;
   copyGradientFlat(calc::grad, g);
   testRequireNear(g, ge, eps);
   session.end();
}

namespace {
// the bath that testMdInit sets, put back to none when the case ends
struct BathReset
{
   ~BathReset()
   {
      bath::kelvin = 0;
      bath::atmsph = 0;
      bath::isothermal = 0;
      bath::isobaric = 0;
   }
};

// the bias inside esum for the method in use (eost.h)
double biasInEsum()
{
   if (use_meta)
      return bgbias;
   if (use_abf)
      return -bostlmda;
   return bgbias - bostlmda;
}

template <class T>
std::vector<T> grabDev(int len, const T* dev)
{
   std::vector<T> v(len);
   darray::copyout(g::q0, len, v.data(), dev);
   waitFor(g::q0);
   return v;
}

std::vector<real> grabUind()
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real(*)[3]>(v.data()), uind);
   waitFor(g::q0);
   return v;
}

// what a lambda move turned down puts back, bit for bit; the energies of the
// terms under analysis are not among them
struct McState
{
   double lmda, g, g1, g2;
   energy_prec dl[7];
   std::vector<virial_prec> v;
   std::vector<grad_prec> gx, gy, gz;

   McState()
      : lmda(lambda)
      , g(cvlam)
      , g1(dcvldlmda)
      , g2(d2cvldlmda2)
      , dl{dedl, d2edl2, decvdl, d2ecvdl2, demdl, depdl, devdl}
      , v(vir, vir + 9)
      , gx(grabDev(n, tinker::gx))
      , gy(grabDev(n, tinker::gy))
      , gz(grabDev(n, tinker::gz))
   {}

   void requireSame(const McState& o) const
   {
      REQUIRE(lmda == o.lmda);
      REQUIRE(g == o.g);
      REQUIRE(g1 == o.g1);
      REQUIRE(g2 == o.g2);
      for (int k = 0; k < 7; ++k) {
         INFO(k);
         REQUIRE(dl[k] == o.dl[k]);
      }
      REQUIRE(v == o.v);
      REQUIRE(gx == o.gx);
      REQUIRE(gy == o.gy);
      REQUIRE(gz == o.gz);
   }
};

// a lambda Monte Carlo move tried at every step of the adaptive bias
// method mode; stochastic dynamics keeps every degree of freedom of a pair
std::string mcKey(const char* mode)
{
   return std::string("lambda 0.5\nlambda-mode ") + mode
      + "\nlambda-propagate mc\nlambda-mc-trial 1\nlambda-mc-move 0.05\nlambda-interval 4\n"
        "integrator stochastic\n";
}
}

TEST_CASE("Colvar-lambda-mc", "[ff][colvar][L2]")
{
   // A lambda Monte Carlo move with the restraint alone and no bias history.
   // A step energy far below any trial turns the move down and leaves the step
   // as it was; far above, the move is taken, and the energy it keeps is that
   // of the restraint at the new lambda, as a fresh energy call finds it.
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   const char* argv[] = {"dummy", kXyz};
   // without analysis, as in dynamics, and with it, where the channel of the
   // restraint keeps its own sums for the turned-down move to put back
   for (int rc : {kRcPlain, kUsage})
      for (const char* mode : {"meta", "ost"}) {
         INFO(mode << " analysis " << (rc == kUsage));
         BathReset bathReset;
         TestLmdaFlagReset lmdaReset;
         TestLmdaFlagGuard guard;
         TestFile fke("", kKey, naclKey(mcKey(mode) + "lambda-cv r 10.0 2.0 4.0\n" + kMap));
         TestSession session(2, argv);
         testMdInit(298., 0.);
         rc_flag = rc;
         session.init();
         REQUIRE(use_lmdamc);
         REQUIRE((use_ost or use_meta));

         // the step: energy, forces, virial and lambda derivatives at 0.5
         const double eps = 1.0e-4;
         energy(calc::v1);
         COMPARE_REALS(esum - biasInEsum(), expectCouple(kNacl, 2.2, 0.5).e, eps);
         if (rc == kUsage)
            COMPARE_REALS(decvdl, expectCouple(kNacl, 2.2, 0.5).dedl, eps);
         const McState before;

         lmdastep = 1;
         esum = -1.0e30;
         lmdaMonte();
         REQUIRE(nlmdamctry == 1);
         REQUIRE(nlmdamcacc == 0);
         REQUIRE(esum == -1.0e30);
         McState().requireSame(before);

         lmdastep = 1;
         esum = 1.0e30;
         lmdaMonte();
         REQUIRE(nlmdamctry == 2);
         REQUIRE(nlmdamcacc == 1);
         REQUIRE(lambda != 0.5);
         REQUIRE(std::fabs(lambda - 0.5) <= 0.05);
         REQUIRE(cvlam == Approx(lambda).epsilon(1.0e-12));
         const auto x = expectCouple(kNacl, 2.2, lambda);
         const double eacc = esum - biasInEsum();
         COMPARE_REALS(eacc, x.e, eps);
         energy(calc::v1);
         COMPARE_REALS(esum - biasInEsum(), eacc, eps);
         COMPARE_REALS(dedl, x.dedl, eps);
         session.end();
      }
}

TEST_CASE("Colvar-lambda-fep", "[ff][colvar][L2]")
{
   // The FEP trial energies at the windows are those of the restraint at the
   // window lambdas, and the native lambda, that of the first window, comes
   // back.
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   TestFile fke("", kKey,
      naclKey("lambda 0.3\nlambda-mode fep\nlambda-window 0.3\nlambda-window 0.5\nlambda-window 0.7\n"
              "lambda-cv r 10.0 2.0 4.0\n"
         + kMap));
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kRcPlain;
   session.init();
   REQUIRE(use_fep);
   REQUIRE(lambda == 0.3);

   const double ltrial[2] = {0.5, 0.7};
   double etrial[2], enative;
   efepTrial(2, ltrial, etrial, enative);
   const double eps = 1.0e-4;
   COMPARE_REALS(etrial[0], expectCouple(kNacl, 2.2, 0.5).e, eps);
   COMPARE_REALS(etrial[1], expectCouple(kNacl, 2.2, 0.7).e, eps);
   COMPARE_REALS(enative, expectCouple(kNacl, 2.2, 0.3).e, eps);
   REQUIRE(lambda == 0.3);
   REQUIRE(cvlam == 0.3);
   session.end();
}

TEST_CASE("Colvar-lambda-full", "[ff][colvar][L2]")
{
   // A geometric lambda with no mutated atoms and the whole AMOEBA water field
   // on, multipoles and mutual polarization with the predictor: next to the
   // same system with no lambda, the restraint adds just its own energy,
   // forces and virial, and the only lambda derivative is its own. A FEP
   // trial and a lambda move turned down leave the step as it was. The
   // analysis buffers keep the lambda derivative of each term apart.
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile fxy("", kXyz, kWater3);
   const char* argv[] = {"dummy", kXyz};
   const std::string base = "parameters water03\na-axis 18.643\nvdw-cutoff 7.0\nmpole-cutoff 6.5\n"
                            "polarization mutual\npolar-eps 0.00001\npolar-predict aspc\n"
                            "cv-atoms pair 1\ncv-atoms pair 4,-5 6\ncv-atoms w3 -7 9\ncv d1 distance pair w3\n";

   double e0;
   std::vector<double> g0;
   std::vector<double> v0(9);
   {
      TestFile fke("", kKey, base);
      TestSession session(2, argv);
      rc_flag = kUsage;
      session.init();
      REQUIRE_FALSE(use(Potent::COLVAR));
      REQUIRE(use(Potent::POLAR));
      energy(calc::v1);
      e0 = esum;
      copyGradientFlat(calc::grad, g0);
      for (int k = 0; k < 9; ++k)
         v0[k] = vir[k];
      session.end();
   }

   BathReset bathReset;
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   TestFile fke("", kKey, base + mcKey("meta") + "lambda-deriv\nlambda-cv d1 5.0 1.0 6.0\n" + kMap);
   TestSession session(2, argv);
   testMdInit(298., 0.);
   rc_flag = kUsage;
   session.init();
   REQUIRE(mutant::nmut == 0);
   REQUIRE(use(Potent::COLVAR));
   REQUIRE(use(Potent::POLAR));

   energy(calc::v1);
   const double estep = esum;
   const auto v = [] {
      ColvarValue c;
      colvarCopyOut(0, c);
      return c;
   }();
   const auto x = expectCouple(kW3, v.q, 0.5);
   const double eps = 1.0e-4;
   COMPARE_REALS(esum - biasInEsum() - e0, x.e, eps);
   std::vector<double> g;
   copyGradientFlat(calc::grad, g);
   auto ge = slotGradient(0, x.de);
   for (size_t k = 0; k < ge.size(); ++k)
      ge[k] += g0[k];
   testRequireNear(g, ge, eps);
   for (int k = 0; k < 9; ++k) {
      INFO(k);
      REQUIRE(vir[k] - v0[k] == Approx(x.de * v.w[k]).margin(eps));
   }
   COMPARE_REALS(dedl, x.dedl, eps);
   COMPARE_REALS(decvdl, x.dedl, eps);
   REQUIRE(demdl == 0);
   REQUIRE(depdl == 0);
   REQUIRE(devdl == 0);

   // a FEP trial at another lambda: the restraint alone changes
   const auto u0 = grabUind();
   const double ltrial = 0.8;
   double etrial, enative;
   efepTrial(1, &ltrial, &etrial, enative);
   COMPARE_REALS(etrial - biasInEsum() - e0, expectCouple(kW3, v.q, 0.8).e, eps);
   COMPARE_REALS(enative, estep, eps);
   REQUIRE(lambda == 0.5);
   const auto u1 = grabUind();
   for (size_t k = 0; k < u0.size(); ++k) {
      INFO(k);
      REQUIRE(u1[k] == Approx(u0[k]).margin(1.0e-5));
   }

   // a lambda move turned down puts back the step
   energy(calc::v1);
   const McState step;
   const auto u2 = grabUind();
   lmdastep = 1;
   esum = -1.0e30;
   lmdaMonte();
   REQUIRE(nlmdamcacc == 0);
   McState().requireSame(step);
   REQUIRE(grabUind() == u2);
   session.end();
}

namespace {
// the numbers of the rows that follow the line holding label in file, with
// Fortran D exponents, up to the first row of another length or a blank line
std::vector<std::vector<double>> readRows(const std::string& file, const std::string& label)
{
   std::ifstream fin(file);
   std::vector<std::vector<double>> rows;
   bool on = false;
   for (std::string line; std::getline(fin, line);) {
      if (not on) {
         on = line.find(label) != std::string::npos;
         continue;
      }
      std::replace(line.begin(), line.end(), 'D', 'E');
      std::istringstream ss(line);
      std::vector<double> row;
      for (double v; ss >> v;)
         row.push_back(v);
      if (row.empty() or (not rows.empty() and row.size() != rows[0].size()))
         break;
      rows.push_back(row);
   }
   return rows;
}
}

TEST_CASE("Colvar-lambda-trajectory", "[ff][colvar][L2]")
{
   // 20 steps of ABF and OST stochastic dynamics of NaCl under the restraint
   // alone, with a fixed random seed, against stored values: the positions,
   // the lambda particle (lambda, theta and its velocity) and the dU/dL that
   // the history keeps at each interval. The restraint reaches the forces, the
   // lambda derivative, the bias and the lambda particle in these few steps;
   // a change in any of them moves the numbers. The stored values come from
   // this code, which the exact free energy of the pair checked: a long TI,
   // ABF and OST run reproduced -kT ln int 4 pi r^2 exp(-E/kT) dr (2026-11-14).
   // The mixed and double precision builds agree to about 1e-8 in the
   // positions and 1e-5 in dU/dL. The OST row keeps the kernel mean force it
   // was stored with (ost-meanforce kernel), which ABF ignores.
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", "cvtraj.xyz");
   TestRemoveFileOnExit farc("cvtraj.arc"), fdyn("cvtraj.dyn"), flog("cvtraj.log");
   struct Ref
   {
      const char* mode;
      double pos[6];
      double lmda[3];    // lambda, theta and its velocity
      double dudl[2][2]; // lambda and dU/dL at steps 10 and 20
   };
   const Ref refs[] = {
      {"abf",
         {0.3147124517834309e-02, 0.1479797042612912e+00, 0.2011461538417913e+00, 0.2418175651593189e+01,
            -0.1881702318345878e+00, -0.1049676748591733e-01},
         {0.3000099142266247e+00, 0.4712634147344069e+00, 0.2276486546435835e-02},
         {{0.3000005951572286e+00, 0.1381219625473022e+02}, {0.3000099142266247e+00, 0.6981115341186523e+01}}},
      {"ost",
         {0.3142616638218754e-02, 0.1479800635908273e+00, 0.2011464132770521e+00, 0.2418178574645047e+01,
            -0.1881704648616453e+00, -0.1049693570574691e-01},
         {0.3000097899701811e+00, 0.4712632195583983e+00, 0.2227980236963097e-02},
         {{0.3000005951572286e+00, 0.1381219625473022e+02}, {0.3000097899701811e+00, 0.6980845928192139e+01}}},
   };
   for (const auto& r : refs) {
      const std::string mode = r.mode, hist = "cvtraj." + mode;
      INFO(mode);
      std::remove(hist.c_str());
      std::remove("cvtraj.dyn");
      std::remove("cvtraj.arc");
      TestRemoveFileOnExit fhist(hist);
      TestFile fke("", "cvtraj.key",
         naclKey("lambda-cv r 10.0 2.0 4.0\n" + kMap
            + "integrator stochastic\nfriction 10.0\nrandomseed 7\nlambda 0.3\nlambda-mode " + mode
            + "\nlambda-dt 0.002\nlambda-interval 10\nost-meanforce kernel\n"));
      const std::string cmd = std::string(TINKER9_BINDIRSTR)
         + "/tinker9 dynamic cvtraj.xyz 20 2.0 0.04 2 298 > cvtraj.log 2>&1";
      const int rc = std::system(cmd.c_str());
      std::ifstream log("cvtraj.log");
      const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
      INFO(text);
      REQUIRE(rc == 0);
      REQUIRE(text.find("Unable to Continue") == std::string::npos);

      const auto pos = readRows("cvtraj.dyn", "Current Atomic Positions");
      REQUIRE(pos.size() == 2);
      for (int k = 0; k < 6; ++k) {
         INFO(k);
         REQUIRE(pos[k / 3].at(k % 3) == Approx(r.pos[k]).margin(1.0e-5));
      }
      const auto lst = readRows(hist, "Lambda State");
      REQUIRE(lst.size() == 1);
      REQUIRE(lst[0].at(0) == Approx(r.lmda[0]).margin(1.0e-8));
      REQUIRE(lst[0].at(1) == Approx(r.lmda[1]).margin(1.0e-8));
      REQUIRE(lst[0].at(2) == Approx(r.lmda[2]).margin(1.0e-6));
      // ABF rows: step, lambda, dU/dL; OST rows: step, seed, lambda, dU/dL, ...
      const auto rows = readRows(hist, mode == "abf" ? "Sample History" : "Gaussian History");
      REQUIRE(rows.size() == 2);
      const int c = mode == "abf" ? 1 : 2;
      for (int i = 0; i < 2; ++i) {
         INFO(i);
         REQUIRE(rows[i].at(0) == 10 * (i + 1));
         REQUIRE(rows[i].at(c) == Approx(r.dudl[i][0]).margin(1.0e-8));
         REQUIRE(rows[i].at(c + 1) == Approx(r.dudl[i][1]).margin(1.0e-3));
      }
   }
}
#endif
