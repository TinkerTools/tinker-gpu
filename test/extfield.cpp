#include "ff/modamoeba.h"
#include "md/integrator.h"
#include "md/pq.h"
#include <tinker/detail/extfld.hh>

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const std::string externalField = "external-field  150  -300  450\n";
static const std::string mpoleOnly = "multipoleterm only\n";
static const std::string polarOnly = "polarizeterm only\n";

TEST_CASE("External-Fields-MPole-Analyze", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField + mpoleOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.1.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial | calc::analyz;
   testBeginWithArgs(argc, argv);
   initialize();

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
   COMPARE_INTS(countReduce(nem), ref_c);

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-Polarize-Analyze", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField + polarOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.2.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial | calc::analyz;
   testBeginWithArgs(argc, argv);
   initialize();

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
   COMPARE_INTS(countReduce(nep), ref_c);

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-MPolar-Analyze", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.3.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial | calc::analyz;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-VdwPchg-Analyze", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/amber.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/amber.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_350df099/amber99sb.prm");
   const char* xn = "amber.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.4.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial | calc::analyz;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-MPole", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField + mpoleOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-Polarize", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField + polarOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-MPolar", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4_Na_shifted.key", "", externalField);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_92528f7d/amoebabio18.prm");
   const char* xn = "water4_Na_shifted.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.3.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

TEST_CASE("External-Fields-VdwPchg", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/amber.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/amber.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_350df099/amber99sb.prm");
   const char* xn = "amber.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.4.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial;
   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}

static const std::string externalFieldFreq = "exfld-freq  20.5\n";
static const std::string integrator = "integrator verlet\n";

TEST_CASE("External-Fields-Dynamic-Freq", "[ff][extfield]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/extfield/water4.dyn");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField + externalFieldFreq + integrator);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";

   // positions and velocities are saved as gradients in the text files
   TestReference rvel(TINKER9_DIRSTR "/test/ref/extfield.5.p.txt");
   TestReference rpos(TINKER9_DIRSTR "/test/ref/extfield.5.q.txt");
   auto ref_pos = rpos.getGradient();
   auto ref_vel = rvel.getGradient();
   auto ref_v = rvel.getVirial();
   const double eps_pos = 0.0001;
   const double eps_vel = 0.001;
   const double eps_vir = 0.001;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE: zero initial velocities
   const double dt_ps = 0.002;
   const int nsteps = 10;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      if (extfld::use_exfld and extfld::use_exfreq) {
         double phs = sin(extfld::exfreq * (istep - 1) * dt_ps);
         for (int i = 0; i < 3; i++) {
            extfld::texfld[i] = phs * extfld::exfld[i];
         }
      }
      vvi.dynamic(istep, dt_ps);
   }

   std::vector<pos_prec> cx(n), cy(n), cz(n);
   std::vector<vel_prec> px(n), py(n), pz(n);
   darray::copyout(g::q0, n, cx.data(), xpos);
   darray::copyout(g::q0, n, cy.data(), ypos);
   darray::copyout(g::q0, n, cz.data(), zpos);
   darray::copyout(g::q0, n, px.data(), vx);
   darray::copyout(g::q0, n, py.data(), vy);
   darray::copyout(g::q0, n, pz.data(), vz);
   waitFor(g::q0);

   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(cx[i], ref_pos[i][0], eps_pos);
      COMPARE_REALS(cy[i], ref_pos[i][1], eps_pos);
      COMPARE_REALS(cz[i], ref_pos[i][2], eps_pos);
   }

   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(px[i], ref_vel[i][0], eps_vel);
      COMPARE_REALS(py[i], ref_vel[i][1], eps_vel);
      COMPARE_REALS(pz[i], ref_vel[i][2], eps_vel);
   }

   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_vir);

   finish();
   testEnd();
}
