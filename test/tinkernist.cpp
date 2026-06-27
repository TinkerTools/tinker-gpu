#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/output.hh>
#include <tinker/routines.h>

#include "test.h"
#include "testrt.h"

#include <unistd.h>

using namespace tinker;

static const std::string nodyn = "nodyn\n";
static const std::string nocoord = "nocoord\n";

TEST_CASE("TinkerNIST-NODYN", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";
   fileExistsAndDelete(fn + ".arc");
   fileExistsAndDelete(fn + ".dyn");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 1;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool dynExists = fileExistsAndDelete(fn + ".dyn");
   REQUIRE(arcExists == true);
   REQUIRE(dynExists == false);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-NOCOORD", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", nocoord);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";
   fileExistsAndDelete(fn + ".arc");
   fileExistsAndDelete(fn + ".dyn");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 1;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool dynExists = fileExistsAndDelete(fn + ".dyn");
   REQUIRE(arcExists == false);
   REQUIRE(dynExists == true);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-DCD", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string dcdkey = "save-ucharge\nsave-ustatic\nsave-uinduce\nsave-udirect\nsave-defield\nsave-"
                                     "tefield\nsave-velocity\ndcd-archive\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", dcdkey);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";
   fileExistsAndDelete(fn + ".dcd");
   fileExistsAndDelete(fn + ".dcdv");
   fileExistsAndDelete(fn + ".dcduc");
   fileExistsAndDelete(fn + ".dcdus");
   fileExistsAndDelete(fn + ".dcdui");
   fileExistsAndDelete(fn + ".dcdud");
   fileExistsAndDelete(fn + ".dcdde");
   fileExistsAndDelete(fn + ".dcdte");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   bool dcdExists = fileExistsAndDelete(fn + ".dcd");
   bool dcdvExists = fileExistsAndDelete(fn + ".dcdv");
   bool dcducExists = fileExistsAndDelete(fn + ".dcduc");
   bool dcdusExists = fileExistsAndDelete(fn + ".dcdus");
   bool dcduiExists = fileExistsAndDelete(fn + ".dcdui");
   bool dcdudExists = fileExistsAndDelete(fn + ".dcdud");
   bool dcddeExists = fileExistsAndDelete(fn + ".dcdde");
   bool dcdteExists = fileExistsAndDelete(fn + ".dcdte");
   REQUIRE(dcdExists == true);
   REQUIRE(dcdvExists == true);
   REQUIRE(dcducExists == true);
   REQUIRE(dcdusExists == true);
   REQUIRE(dcduiExists == true);
   REQUIRE(dcdudExists == true);
   REQUIRE(dcddeExists == true);
   REQUIRE(dcdteExists == true);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-SAVE1", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string
      savekey = "save-ucharge\nsave-ustatic\nsave-uinduce\nsave-udirect\nsave-defield\nsave-tefield\nsave-velocity\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", savekey + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";
   fileExistsAndDelete(fn + ".arc");
   fileExistsAndDelete(fn + ".vel");
   fileExistsAndDelete(fn + ".uchg");
   fileExistsAndDelete(fn + ".ustc");
   fileExistsAndDelete(fn + ".uind");
   fileExistsAndDelete(fn + ".udir");
   fileExistsAndDelete(fn + ".def");
   fileExistsAndDelete(fn + ".tef");

   auto arc_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.arc");
   auto vel_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.vel");
   auto uchg_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.uchg");
   auto ustc_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.ustc");
   auto uind_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.uind");
   auto udir_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.udir");
   auto def_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.def");
   auto tef_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.tef");
   const double eps_arc = 0.001;
   const double eps_vel = 0.002;
   const double eps_uchg = 0.001;
   const double eps_ustc = 0.001;
   const double eps_uind = 0.001;
   const double eps_udir = 0.001;
   const double eps_def = 0.001;
   const double eps_tef = 0.001;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   auto arc_tst = readAmoebaCoordinateFile(fn + ".arc");
   auto vel_tst = readAmoebaCoordinateFile(fn + ".vel");
   auto uchg_tst = readAmoebaCoordinateFile(fn + ".uchg");
   auto ustc_tst = readAmoebaCoordinateFile(fn + ".ustc");
   auto uind_tst = readAmoebaCoordinateFile(fn + ".uind");
   auto udir_tst = readAmoebaCoordinateFile(fn + ".udir");
   auto def_tst = readAmoebaCoordinateFile(fn + ".def");
   auto tef_tst = readAmoebaCoordinateFile(fn + ".tef");

   for (int istep = 0; istep < nsteps; ++istep) {
      /// arc
      auto compare_arc = [&](int i) {
         COMPARE_REALS(arc_tst[istep][i].x, arc_ref[istep][i].x, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].y, arc_ref[istep][i].y, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].z, arc_ref[istep][i].z, eps_arc);
      };
      /// vel
      auto compare_vel = [&](int i) {
         COMPARE_REALS(vel_tst[istep][i].x, vel_ref[istep][i].x, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].y, vel_ref[istep][i].y, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].z, vel_ref[istep][i].z, eps_vel);
      };
      /// uchg
      auto compare_uchg = [&](int i) {
         COMPARE_REALS(uchg_tst[istep][i].x, uchg_ref[istep][i].x, eps_uchg);
         COMPARE_REALS(uchg_tst[istep][i].y, uchg_ref[istep][i].y, eps_uchg);
         COMPARE_REALS(uchg_tst[istep][i].z, uchg_ref[istep][i].z, eps_uchg);
      };
      /// ustc
      auto compare_ustc = [&](int i) {
         COMPARE_REALS(ustc_tst[istep][i].x, ustc_ref[istep][i].x, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].y, ustc_ref[istep][i].y, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].z, ustc_ref[istep][i].z, eps_ustc);
      };
      /// uind
      auto compare_uind = [&](int i) {
         COMPARE_REALS(uind_tst[istep][i].x, uind_ref[istep][i].x, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].y, uind_ref[istep][i].y, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].z, uind_ref[istep][i].z, eps_uind);
      };
      /// udir
      auto compare_udir = [&](int i) {
         COMPARE_REALS(udir_tst[istep][i].x, udir_ref[istep][i].x, eps_udir);
         COMPARE_REALS(udir_tst[istep][i].y, udir_ref[istep][i].y, eps_udir);
         COMPARE_REALS(udir_tst[istep][i].z, udir_ref[istep][i].z, eps_udir);
      };
      /// def
      auto compare_def = [&](int i) {
         COMPARE_REALS(def_tst[istep][i].x, def_ref[istep][i].x, eps_def);
         COMPARE_REALS(def_tst[istep][i].y, def_ref[istep][i].y, eps_def);
         COMPARE_REALS(def_tst[istep][i].z, def_ref[istep][i].z, eps_def);
      };
      /// tef
      auto compare_tef = [&](int i) {
         COMPARE_REALS(tef_tst[istep][i].x, tef_ref[istep][i].x, eps_tef);
         COMPARE_REALS(tef_tst[istep][i].y, tef_ref[istep][i].y, eps_tef);
         COMPARE_REALS(tef_tst[istep][i].z, tef_ref[istep][i].z, eps_tef);
      };
      // test the first j and last j atoms
      int j = 2;
      for (int i = 0; i < j; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_uchg(i);
         compare_ustc(i);
         compare_uind(i);
         compare_udir(i);
         compare_def(i);
         compare_tef(i);
      }
      for (int i = n - j; i < n; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_uchg(i);
         compare_ustc(i);
         compare_uind(i);
         compare_udir(i);
         compare_def(i);
         compare_tef(i);
      }
   }

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool velExists = fileExistsAndDelete(fn + ".vel");
   bool uchgExists = fileExistsAndDelete(fn + ".uchg");
   bool ustcExists = fileExistsAndDelete(fn + ".ustc");
   bool uindExists = fileExistsAndDelete(fn + ".uind");
   bool udirExists = fileExistsAndDelete(fn + ".udir");
   bool defExists = fileExistsAndDelete(fn + ".def");
   bool tefExists = fileExistsAndDelete(fn + ".tef");
   REQUIRE(arcExists == true);
   REQUIRE(velExists == true);
   REQUIRE(uchgExists == true);
   REQUIRE(ustcExists == true);
   REQUIRE(uindExists == true);
   REQUIRE(udirExists == true);
   REQUIRE(defExists == true);
   REQUIRE(tefExists == true);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-SAVE2", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.dyn");
   static const std::string
      savekey = "save-ucharge\nsave-ustatic\nsave-uinduce\nsave-udirect\nsave-defield\nsave-tefield\nsave-velocity\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.key", "", savekey + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_aad9340c/water21.prm");
   const char* xn = "water30_2.xyz";
   static const std::string fn = "water30_2";
   fileExistsAndDelete(fn + ".arc");
   fileExistsAndDelete(fn + ".vel");
   fileExistsAndDelete(fn + ".uchg");
   fileExistsAndDelete(fn + ".ustc");
   fileExistsAndDelete(fn + ".uind");
   fileExistsAndDelete(fn + ".udir");
   fileExistsAndDelete(fn + ".def");
   fileExistsAndDelete(fn + ".tef");

   auto arc_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.arc");
   auto vel_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.vel");
   auto uchg_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.uchg");
   auto ustc_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.ustc");
   auto uind_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.uind");
   auto udir_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.udir");
   auto def_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.def");
   auto tef_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist_2.tef");
   const double eps_arc = 0.001;
   const double eps_vel = 0.002;
   const double eps_uchg = 0.001;
   const double eps_ustc = 0.001;
   const double eps_uind = 0.001;
   const double eps_udir = 0.001;
   const double eps_def = 0.001;
   const double eps_tef = 0.001;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   auto arc_tst = readAmoebaCoordinateFile(fn + ".arc");
   auto vel_tst = readAmoebaCoordinateFile(fn + ".vel");
   auto uchg_tst = readAmoebaCoordinateFile(fn + ".uchg");
   auto ustc_tst = readAmoebaCoordinateFile(fn + ".ustc");
   auto uind_tst = readAmoebaCoordinateFile(fn + ".uind");
   auto udir_tst = readAmoebaCoordinateFile(fn + ".udir");
   auto def_tst = readAmoebaCoordinateFile(fn + ".def");
   auto tef_tst = readAmoebaCoordinateFile(fn + ".tef");

   for (int istep = 0; istep < nsteps; ++istep) {
      /// arc
      auto compare_arc = [&](int i) {
         COMPARE_REALS(arc_tst[istep][i].x, arc_ref[istep][i].x, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].y, arc_ref[istep][i].y, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].z, arc_ref[istep][i].z, eps_arc);
      };
      /// vel
      auto compare_vel = [&](int i) {
         COMPARE_REALS(vel_tst[istep][i].x, vel_ref[istep][i].x, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].y, vel_ref[istep][i].y, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].z, vel_ref[istep][i].z, eps_vel);
      };
      /// uchg
      auto compare_uchg = [&](int i) {
         COMPARE_REALS(uchg_tst[istep][i].x, uchg_ref[istep][i].x, eps_uchg);
         COMPARE_REALS(uchg_tst[istep][i].y, uchg_ref[istep][i].y, eps_uchg);
         COMPARE_REALS(uchg_tst[istep][i].z, uchg_ref[istep][i].z, eps_uchg);
      };
      /// ustc
      auto compare_ustc = [&](int i) {
         COMPARE_REALS(ustc_tst[istep][i].x, ustc_ref[istep][i].x, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].y, ustc_ref[istep][i].y, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].z, ustc_ref[istep][i].z, eps_ustc);
      };
      /// uind
      auto compare_uind = [&](int i) {
         COMPARE_REALS(uind_tst[istep][i].x, uind_ref[istep][i].x, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].y, uind_ref[istep][i].y, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].z, uind_ref[istep][i].z, eps_uind);
      };
      /// udir
      auto compare_udir = [&](int i) {
         COMPARE_REALS(udir_tst[istep][i].x, udir_ref[istep][i].x, eps_udir);
         COMPARE_REALS(udir_tst[istep][i].y, udir_ref[istep][i].y, eps_udir);
         COMPARE_REALS(udir_tst[istep][i].z, udir_ref[istep][i].z, eps_udir);
      };
      /// def
      auto compare_def = [&](int i) {
         COMPARE_REALS(def_tst[istep][i].x, def_ref[istep][i].x, eps_def);
         COMPARE_REALS(def_tst[istep][i].y, def_ref[istep][i].y, eps_def);
         COMPARE_REALS(def_tst[istep][i].z, def_ref[istep][i].z, eps_def);
      };
      /// tef
      auto compare_tef = [&](int i) {
         COMPARE_REALS(tef_tst[istep][i].x, tef_ref[istep][i].x, eps_tef);
         COMPARE_REALS(tef_tst[istep][i].y, tef_ref[istep][i].y, eps_tef);
         COMPARE_REALS(tef_tst[istep][i].z, tef_ref[istep][i].z, eps_tef);
      };
      // test the first j and last j atoms
      int j = 2;
      for (int i = 0; i < j; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_uchg(i);
         compare_ustc(i);
         compare_uind(i);
         compare_udir(i);
         compare_def(i);
         compare_tef(i);
      }
      for (int i = n - j; i < n; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_uchg(i);
         compare_ustc(i);
         compare_uind(i);
         compare_udir(i);
         compare_def(i);
         compare_tef(i);
      }
   }

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool velExists = fileExistsAndDelete(fn + ".vel");
   bool uchgExists = fileExistsAndDelete(fn + ".uchg");
   bool ustcExists = fileExistsAndDelete(fn + ".ustc");
   bool uindExists = fileExistsAndDelete(fn + ".uind");
   bool udirExists = fileExistsAndDelete(fn + ".udir");
   bool defExists = fileExistsAndDelete(fn + ".def");
   bool tefExists = fileExistsAndDelete(fn + ".tef");
   REQUIRE(arcExists == true);
   REQUIRE(velExists == true);
   REQUIRE(uchgExists == true);
   REQUIRE(ustcExists == true);
   REQUIRE(uindExists == true);
   REQUIRE(udirExists == true);
   REQUIRE(defExists == true);
   REQUIRE(tefExists == true);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-SAVE-ONLY", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string savekey = "save-only -4 "
                                      "102\nsave-ucharge\nsave-ustatic\nsave-uinduce\nsave-udirect\nsave-defield\nsave-"
                                      "tefield\nsave-velocity\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", savekey + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";
   fileExistsAndDelete(fn + ".arc");
   fileExistsAndDelete(fn + ".vel");
   fileExistsAndDelete(fn + ".uchg");
   fileExistsAndDelete(fn + ".ustc");
   fileExistsAndDelete(fn + ".uind");
   fileExistsAndDelete(fn + ".udir");
   fileExistsAndDelete(fn + ".def");
   fileExistsAndDelete(fn + ".tef");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   auto arc_tst = readAmoebaCoordinateFile(fn + ".arc");
   auto vel_tst = readAmoebaCoordinateFile(fn + ".vel");
   auto uchg_tst = readAmoebaCoordinateFile(fn + ".uchg");
   auto ustc_tst = readAmoebaCoordinateFile(fn + ".ustc");
   auto uind_tst = readAmoebaCoordinateFile(fn + ".uind");
   auto udir_tst = readAmoebaCoordinateFile(fn + ".udir");
   auto def_tst = readAmoebaCoordinateFile(fn + ".def");
   auto tef_tst = readAmoebaCoordinateFile(fn + ".tef");

   for (int istep = 0; istep < nsteps; ++istep) {
      REQUIRE(arc_tst[0].size() == 99);
      REQUIRE(vel_tst[0].size() == 99);
      REQUIRE(uchg_tst[0].size() == 99);
      REQUIRE(ustc_tst[0].size() == 99);
      REQUIRE(uind_tst[0].size() == 99);
      REQUIRE(udir_tst[0].size() == 99);
      REQUIRE(def_tst[0].size() == 99);
      REQUIRE(tef_tst[0].size() == 99);
   }

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool velExists = fileExistsAndDelete(fn + ".vel");
   bool uchgExists = fileExistsAndDelete(fn + ".uchg");
   bool ustcExists = fileExistsAndDelete(fn + ".ustc");
   bool uindExists = fileExistsAndDelete(fn + ".uind");
   bool udirExists = fileExistsAndDelete(fn + ".udir");
   bool defExists = fileExistsAndDelete(fn + ".def");
   bool tefExists = fileExistsAndDelete(fn + ".tef");
   REQUIRE(arcExists == true);
   REQUIRE(velExists == true);
   REQUIRE(uchgExists == true);
   REQUIRE(ustcExists == true);
   REQUIRE(uindExists == true);
   REQUIRE(udirExists == true);
   REQUIRE(defExists == true);
   REQUIRE(tefExists == true);

   finish();
   testEnd();
}

std::vector<std::array<double, 3>> read_logfile_1(const std::string& filename, const std::string& label)
{
   std::vector<std::array<double, 3>> vectors;
   std::ifstream file(filename);
   std::string line;
   while (std::getline(file, line)) {
      if (line.find(label) != std::string::npos) {
         std::istringstream iss(line);
         std::string tmp;
         double x, y, z;
         iss >> tmp >> tmp >> tmp >> x >> y >> z;
         vectors.push_back({x, y, z});
      }
   }
   return vectors;
}

std::vector<std::array<double, 3>> read_logfile_2(const std::string& filename, const std::string& label)
{
   std::vector<std::array<double, 3>> vectors;
   std::ifstream file(filename);
   std::string line;
   while (std::getline(file, line)) {
      if (line.find(label) != std::string::npos) {
         // Skip the column header
         std::getline(file, line);
         for (int k = 0; k < 4; ++k) {
            std::getline(file, line);
            std::istringstream iss(line);
            int tmp;
            double x, y, z;
            iss >> tmp >> x >> y >> z;
            vectors.push_back({x, y, z});
         }
      }
   }
   return vectors;
}

TEST_CASE("TinkerNIST-SAVE-SYSTEM", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string savekey = "save-usystem\nsave-vsystem\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", savekey + nocoord + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";

   const double eps_u = 0.0001;
   const double eps_v = 0.01;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   int saved_stdout_fd = dup(fileno(stdout));
   if (saved_stdout_fd == -1) {
      throw std::runtime_error("Failed to duplicate stdout");
   }
   static const std::string outputfn = "output.log";
   FILE* out = freopen(outputfn.c_str(), "w", stdout);
   if (!out) {
      close(saved_stdout_fd);
      throw std::runtime_error("Failed to redirect stdout to " + outputfn);
   }

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   finish();
   testEnd();

   // run after testEnd(); to flush fortran buffer
   fflush(stdout);
   dup2(saved_stdout_fd, fileno(stdout));
   close(saved_stdout_fd);

   std::vector<std::array<double, 3>> uchg_ref;
   std::vector<std::array<double, 3>> ustc_ref;
   std::vector<std::array<double, 3>> uind_ref;
   std::vector<std::array<double, 3>> auchg_ref;
   std::vector<std::array<double, 3>> austc_ref;
   std::vector<std::array<double, 3>> auind_ref;
   std::vector<std::array<double, 3>> avelo_ref;
   uchg_ref.push_back({6.892510, -27.773255, 75.057906});
   uchg_ref.push_back({6.828811, -27.794699, 74.920880});
   ustc_ref.push_back({-7.366166, -7.871300, 6.450486});
   ustc_ref.push_back({-7.348372, -7.888462, 6.419733});
   uind_ref.push_back({-26.743482, -16.250646, 4.786944});
   uind_ref.push_back({-26.846484, -16.328934, 4.642692});
   auchg_ref.push_back({303.701119, 230.038309, 292.209284});
   auchg_ref.push_back({-331.940017, -259.883598, -267.693160});
   auchg_ref.push_back({203.333866, 139.690946, 202.363634});
   auchg_ref.push_back({-168.202458, -137.618912, -151.821853});
   auchg_ref.push_back({303.710240, 230.037088, 292.181659});
   auchg_ref.push_back({-331.966875, -259.926399, -267.830151});
   auchg_ref.push_back({203.312572, 139.706841, 202.374912});
   auchg_ref.push_back({-168.227126, -137.612229, -151.805540});
   austc_ref.push_back({-6.783206, -7.362798, 6.004690});
   austc_ref.push_back({-0.582960, -0.508502, 0.445796});
   austc_ref.push_back({0.000000, 0.000000, 0.000000});
   austc_ref.push_back({0.000000, 0.000000, 0.000000});
   austc_ref.push_back({-6.766177, -7.383621, 5.971236});
   austc_ref.push_back({-0.582195, -0.504841, 0.448497});
   austc_ref.push_back({0.000000, 0.000000, 0.000000});
   austc_ref.push_back({0.000000, 0.000000, 0.000000});
   auind_ref.push_back({-13.626743, -10.888829, 4.352743});
   auind_ref.push_back({-11.936671, -7.998449, 2.410177});
   auind_ref.push_back({0.049248, -0.011254, -0.005858});
   auind_ref.push_back({-1.229316, 2.647885, -1.970117});
   auind_ref.push_back({-13.665637, -10.910517, 4.225972});
   auind_ref.push_back({-11.981849, -8.039010, 2.342266});
   auind_ref.push_back({0.048904, -0.010808, -0.005464});
   auind_ref.push_back({-1.247902, 2.631401, -1.920081});
   avelo_ref.push_back({-0.245765, 3.852261, 13.515902});
   avelo_ref.push_back({-9.901128, -29.294194, -93.776943});
   avelo_ref.push_back({-4.357503, 3.194936, 2.389992});
   avelo_ref.push_back({5.175468, -1.274002, -3.399699});
   avelo_ref.push_back({-0.326293, 3.941636, 14.252432});
   avelo_ref.push_back({-4.246289, -28.927602, -104.687128});
   avelo_ref.push_back({-4.468945, 3.446065, 2.327113});
   avelo_ref.push_back({5.131407, -1.477522, -3.369982});

   auto uchg_tst = read_logfile_1(outputfn, " System Charge Dipole");
   auto ustc_tst = read_logfile_1(outputfn, " System Static Dipole");
   auto uind_tst = read_logfile_1(outputfn, " System Induced Dipole");
   auto auchg_tst = read_logfile_2(outputfn, " Charge Dipole by Atom Type:");
   auto austc_tst = read_logfile_2(outputfn, " Static Dipole by Atom Type:");
   auto auind_tst = read_logfile_2(outputfn, " Induced Dipole by Atom Type:");
   auto avelo_tst = read_logfile_2(outputfn, " Velocity by Atom Type:");

   fileExistsAndDelete(outputfn);

   for (int istep = 0; istep < nsteps; ++istep) {
      for (int i = 0; i < 3; ++i) {
         COMPARE_REALS(uchg_ref[istep][i], uchg_tst[istep][i], eps_u);
         COMPARE_REALS(ustc_ref[istep][i], ustc_tst[istep][i], eps_u);
         COMPARE_REALS(uind_ref[istep][i], uind_tst[istep][i], eps_u);
      }
   }

   for (int istep = 0; istep < nsteps * 4; ++istep) {
      for (int i = 0; i < 3; ++i) {
         COMPARE_REALS(auchg_ref[istep][i], auchg_tst[istep][i], eps_u);
         COMPARE_REALS(austc_ref[istep][i], austc_tst[istep][i], eps_u);
         COMPARE_REALS(auind_ref[istep][i], auind_tst[istep][i], eps_u);
         COMPARE_REALS(avelo_ref[istep][i], avelo_tst[istep][i], eps_v);
      }
   }
}

TEST_CASE("TinkerNIST-EXC-MOMENT", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string savekey = "exc-moment -2677 2684\nsave-usystem\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", savekey + nocoord + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";

   const double eps_u = 0.0001;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   int saved_stdout_fd = dup(fileno(stdout));
   if (saved_stdout_fd == -1) {
      throw std::runtime_error("Failed to duplicate stdout");
   }
   static const std::string outputfn = "output.log";
   FILE* out = freopen(outputfn.c_str(), "w", stdout);
   if (!out) {
      close(saved_stdout_fd);
      throw std::runtime_error("Failed to redirect stdout to " + outputfn);
   }

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   finish();
   testEnd();

   // run after testEnd(); to flush fortran buffer
   fflush(stdout);
   dup2(saved_stdout_fd, fileno(stdout));
   close(saved_stdout_fd);

   std::vector<std::array<double, 3>> uchg_ref;
   std::vector<std::array<double, 3>> ustc_ref;
   std::vector<std::array<double, 3>> uind_ref;
   uchg_ref.push_back({-28.238899, -29.845290, 24.516125});
   uchg_ref.push_back({-28.256635, -29.889311, 24.351508});
   ustc_ref.push_back({-7.366166, -7.871300, 6.450486});
   ustc_ref.push_back({-7.348372, -7.888462, 6.419733});
   uind_ref.push_back({-25.563414, -18.887278, 6.762920});
   uind_ref.push_back({-25.647485, -18.949528, 6.568238});

   auto uchg_tst = read_logfile_1(outputfn, " System Charge Dipole");
   auto ustc_tst = read_logfile_1(outputfn, " System Static Dipole");
   auto uind_tst = read_logfile_1(outputfn, " System Induced Dipole");

   fileExistsAndDelete(outputfn);

   for (int istep = 0; istep < nsteps; ++istep) {
      for (int i = 0; i < 3; ++i) {
         COMPARE_REALS(uchg_ref[istep][i], uchg_tst[istep][i], eps_u);
         COMPARE_REALS(ustc_ref[istep][i], ustc_tst[istep][i], eps_u);
         COMPARE_REALS(uind_ref[istep][i], uind_tst[istep][i], eps_u);
      }
   }
}
