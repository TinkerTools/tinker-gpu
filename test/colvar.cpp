#include "ff/colvar.h"
#include "ff/atom.h"
#include "tool/darray.h"

#include "test.h"
#include "testrt.h"

#include <cmath>
#include <string>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/molcul.hh>
#include <vector>

using namespace tinker;

namespace {
const char* kXyz = "test_colvar.xyz";
const char* kKey = "test_colvar.key";

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

const int kUsage = calc::xyz | calc::mass | calc::vmask;
}

TEST_CASE("Colvar-distance-pair", "[ff][colvar][L1]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fxy(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", kXyz);
   TestFile fke("", kKey,
      "parameters amoeba09\n"
      "vdwterm only\n"
      "CV-Atoms  Na  1\n"
      "cv-atoms  cl  2\n"
      "cv  r  Distance  na  CL\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   REQUIRE(colvarCount() == 1);
   REQUIRE(colvarFind("R") == 0);
   REQUIRE(colvarFind("r") == 0);
   REQUIRE(colvarFind("q") == -1);
   REQUIRE(colvarName(0) == "R");

   colvarEval();
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
      "vdwterm only\n"
      "neighbor-list\n"
      "vdw-cutoff 7.0\n"
      "a-axis 18.643\n"
      "cv-atoms  na  1\n"
      "cv-atoms  cl  2\n"
      "cv  r  distance  na  cl\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   colvarEval();
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
   TestFile fxy("", kXyz,
      "     9  Water\n"
      "     1  O     -2.037380    1.597451   -3.481484     1     2     3\n"
      "     2  H     -1.125125    1.412994   -3.213200     2     1\n"
      "     3  H     -2.541095    1.754841   -2.666805     2     1\n"
      "     4  O     17.790469   -0.894103    0.301909     1     5     6\n"
      "     5  H     17.534545   -1.104566   -0.607506     2     4\n"
      "     6  H     18.112422    0.011819    0.268479     2     4\n"
      "     7  O      1.259139   -2.415289    1.790163     1     8     9\n"
      "     8  H      1.999646   -1.796414    1.703920     2     7\n"
      "     9  H      0.543913   -2.030812    1.270042     2     7\n");
   TestFile fke("", kKey,
      "parameters water03\n"
      "vdwterm only\n"
      "neighbor-list\n"
      "vdw-cutoff 7.0\n"
      "a-axis 18.643\n"
      "cv-atoms  pair  1\n"
      "cv-atoms  pair  4,-5 6\n"
      "cv-atoms  w3    -7 9\n"
      "cv-atoms  h     2 3\n"
      "cv-atoms  odd   1 4 5 8\n"
      "cv  d1  distance  pair  w3\n"
      "cv  d2  distance  h     w3\n"
      "cv  d3  distance  odd   h\n");
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   REQUIRE(colvarCount() == 3);
   colvarEval();
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

   auto refused = [&](const std::string& keys, const char* message) {
      INFO(keys);
      TestFile fke("", kKey, "parameters amoeba09\nvdwterm only\n" + keys);
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
   refused("cv-atoms na x\n", "Cannot Read the Atoms of Set NA");
   refused("cv-atoms na -2 1\n", "Atom Range 2 to 1 of Set NA Is Empty");

   // the centers of the sets need the atomic masses on the device
   {
      TestFile fke("", kKey, "parameters amoeba09\nvdwterm only\n" + sets + "cv r distance na cl\n");
      TestSession session(2, argv);
      rc_flag = calc::xyz | calc::vmask;
      REQUIRE_THROWS_WITH(session.init(), Catch::Matchers::ContainsSubstring("Atomic Masses the CVs Need"));
   }

   // atom sets alone define no CV
   TestFile fke("", kKey, "parameters amoeba09\nvdwterm only\n" + sets);
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();
   REQUIRE(colvarCount() == 0);
   colvarEval();
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
   std::string key = "parameters water03\nrestrainterm only\ncv-atoms a -1 300\ncv-atoms o 2700\n";
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
   key += "cv big distance a b\ncv o distance b o\n";
   TestFile fke("", kKey, key);
   const char* argv[] = {"dummy", kXyz};
   TestSession session(2, argv);
   rc_flag = kUsage;
   session.init();

   std::vector<int> sa;
   for (int k = 0; k < 300; ++k)
      sa.push_back(k);
   colvarEval();
   const double eps = 1.0e-4;
   compareCv("big", refDistance(sa, sb, a), eps);
   compareCv("o", refDistance(sb, {2699}, a), eps);

   // a second evaluation starts the sums from zero again
   colvarEval();
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
      "restrainterm only\n"
      "cv-atoms  lig   10 11 12\n"
      "cv-atoms  pock  3 7\n"
      "cv-atoms  na    20\n"
      "cv  d1  distance  lig  pock\n"
      "cv  d2  distance  na   lig\n");
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

   colvarEval();
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
