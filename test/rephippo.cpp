#include "ff/atom.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/erepel.h"
#include "ff/potent.h"
#include "tool/darray.h"

#include "test.h"
#include "testrt.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace tinker;

TEST_CASE("EREPEL-1", "[ff][hippo][erepel][c5h12-acnh2]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/repuls/rep.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "rep.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/rephippo.1.txt");
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
   COMPARE_INTS(countReduce(nrep), ref_c);

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

TEST_CASE("EREPEL-2", "[ff][hippo][erepel][water]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/water10/h2o10.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/repuls/repwater.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/water19.prm");
   const char* xn = "h2o10.xyz";
   const char* kn = "repwater.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/rephippo.2.txt");
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
   COMPARE_INTS(countReduce(nrep), ref_c);

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

// The soft core scales the repulsion of a ligand by a weight that depends on
// the distance, so the gradient has a term of its own, one that vanishes at
// both ends of the vdw lambda. Part way along, the analytic gradient on the
// ligand atoms is held to central differences of the energy.
TEST_CASE("EREPEL-3-SOFTCORE", "[ff][hippo][erepel][dmso]")
{
   TestLmdaFlagReset lmdaReset;
   TestFile fx1(TINKER9_DIRSTR "/test/file/dmso/dmso.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/fep/hippo.key", "repsoft.key", "\nrepulsionterm only\nlambda 0.9\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* argv[] = {"dummy", "dmso.xyz", "-k", "repsoft.key"};
   int argc = 4;

   // The energy is summed in fixed point from single precision pair energies,
   // which leaves a few 1e-3 of noise in a difference over this step; the
   // gradient term that was once scaled twice is off by 2e-2 here.
   const double h = 2.0e-3;
   const double eps = testGetEps(6.0e-3, 3.0e-4);
   const int nlig = 10; // the ligand is the first molecule

   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad;
   session.init();
   REQUIRE(use(Potent::REPULS));
   COMPARE_REALS(vlam, 0.9, 1.0e-6);

   energy(calc::v4);
   std::vector<double> g[3];
   for (auto& v : g)
      v.resize(n);
   copyGradient(calc::grad, g[0].data(), g[1].data(), g[2].data());

   std::vector<pos_prec> x0[3];
   pos_prec* const pos[3] = {xpos, ypos, zpos};
   for (int a = 0; a < 3; ++a) {
      x0[a].resize(n);
      darray::copyout(g::q0, n, x0[a].data(), pos[a]);
   }
   waitFor(g::q0);
   auto energyAt = [&](int i, int a, double d) {
      std::vector<pos_prec> x1 = x0[a];
      x1[i] += d;
      darray::copyin(g::q0, n, pos[a], x1.data());
      waitFor(g::q0);
      copyPosToXyz(true);
      energy(calc::v0);
      double e = esum;
      darray::copyin(g::q0, n, pos[a], x0[a].data());
      waitFor(g::q0);
      return e;
   };

   double worst = 0, largest = 0;
   for (int i = 0; i < nlig; ++i) {
      for (int a = 0; a < 3; ++a) {
         double numer = (energyAt(i, a, h) - energyAt(i, a, -h)) / (2 * h);
         worst = std::max(worst, std::fabs(g[a][i] - numer));
         largest = std::max(largest, std::fabs(numer));
      }
   }
   copyPosToXyz(true);
   REQUIRE(largest > 1.0);
   REQUIRE(worst < eps);

   session.end();
}
