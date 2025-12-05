#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const std::string externalField = "external-field  150  -300  450\n";
static const std::string mpoleOnly = "multipoleterm only\n";
static const std::string polarOnly = "polarizeterm only\n";

TEST_CASE("External-Fields-MPole-Analyze", "[ff][extfield]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField + mpoleOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
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

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial | calc::analyz;
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
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField + polarOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
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

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial | calc::analyz;
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
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.3.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial | calc::analyz;
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

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial | calc::analyz;
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
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField + mpoleOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
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
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField + polarOnly);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
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
   TestFile fx1(TINKER9_DIRSTR "/test/file/extfield/water4.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/extfield/water4.key", "", externalField);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   const char* xn = "water4.xyz";
   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/extfield.3.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
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

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
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
