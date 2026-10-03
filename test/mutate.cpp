#include "ff/amoeba/emplar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/mpole.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"
#include "tool/platform.h"

#include <tinker/detail/atoms.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace tinker;

// The fixtures that need neither lambda derivatives nor dual topology (001-029
// and most of 089-103) run on the OpenACC build's own kernels too; the rest of
// this file is CUDA only.
#if TINKER_GPULANG_CUDA || TINKER_GPULANG_OPENACC

namespace {
/// Tolerances a fixture overrides. Each check left unset keeps the default in
/// runFixture; one that is set takes the given single/mixed and double
/// precision values instead. Rows set them by name, e.g. Tols().virial(...).
struct Tols
{
   struct Tol
   {
      double mixed = 0, dbl = 0;
   };
   Tol e, g, v, l, l2;

   Tols energy(double mixed, double dbl) const { return with(&Tols::e, mixed, dbl); }
   Tols grad(double mixed, double dbl) const { return with(&Tols::g, mixed, dbl); }
   Tols virial(double mixed, double dbl) const { return with(&Tols::v, mixed, dbl); }
   Tols lmda(double mixed, double dbl) const { return with(&Tols::l, mixed, dbl); }
   Tols lmda2(double mixed, double dbl) const { return with(&Tols::l2, mixed, dbl); }

   /// The tolerance of a check: the override if set, otherwise the default.
   static double pick(const Tol& t, double mixed, double dbl)
   {
      return (t.mixed or t.dbl) ? testGetEps(t.mixed, t.dbl) : testGetEps(mixed, dbl);
   }

private:
   Tols with(Tol Tols::*m, double mixed, double dbl) const
   {
      Tols t = *this;
      t.*m = {mixed, dbl};
      return t;
   }
};

struct Fixture
{
   const char* name;
   const char* base;
   bool checkm, checkp, checkv, dolmda;
   const char* ref = nullptr; ///< Reference borrowed from another fixture, if any.
   Tols tol = {};             ///< Tolerances that differ from the defaults.
};

const Fixture kFixtures[] = {
   {"001_water_ye_m10", "water", true, true, false, false},
   {"002_water_ne_m10", "water", true, true, false, false},
   {"003_water_ye_m05", "water", true, true, false, false},
   {"004_water_ne_m05", "water", true, true, false, false},
   {"005_water_ye_m00", "water", true, true, false, false},
   {"006_water_ne_m00", "water", true, true, false, false},
   {"007_water_v10", "water", false, false, true, false},
   {"008_water_v05", "water", false, false, true, false},
   {"009_water_v00", "water", false, false, true, false},
   {"010_water_ye_m10", "water", true, false, false, false},
   {"011_water_ne_m10", "water", true, false, false, false},
   {"012_water_ye_m05", "water", true, false, false, false},
   {"013_water_ne_m05", "water", true, false, false, false},
   {"014_water_ye_m00", "water", true, false, false, false},
   {"015_water_ne_m00", "water", true, false, false, false},
   {"016_water_ye_p10", "water", false, true, false, false},
   {"017_water_ne_p10", "water", false, true, false, false},
   {"018_water_ye_p05", "water", false, true, false, false},
   {"019_water_ne_p05", "water", false, true, false, false},
   {"020_water_ye_p00", "water", false, true, false, false},
   {"021_water_ne_p00", "water", false, true, false, false},
   {"022_water_ye_m10p05", "water", true, true, false, false},
   {"023_water_ne_m10p05", "water", true, true, false, false},
   {"024_water_ye_m05p10", "water", true, true, false, false},
   {"025_water_ne_m05p10", "water", true, true, false, false},
   {"026_water_ye_m05p00", "water", true, true, false, false},
   {"027_water_ne_m05p00", "water", true, true, false, false},
   {"028_water_ye_m00p05", "water", true, true, false, false},
   {"029_water_ne_m00p05", "water", true, true, false, false},
   {"030_water_ast_ye_m10", "water2", true, false, false, true},
   {"031_water_ast_ne_m10", "water2", true, false, false, true},
   {"032_water_ast_ye_m05", "water2", true, false, false, true},
   {"033_water_ast_ne_m05", "water2", true, false, false, true},
   {"034_water_ast_ye_m00", "water2", true, false, false, true},
   {"035_water_ast_ne_m00", "water2", true, false, false, true},
   {"036_water_ast_v10", "water2", false, false, true, true},
   {"037_water_ast_v05", "water2", false, false, true, true},
   {"038_water_ast_v00", "water2", false, false, true, true},
   {"039_water_ast_ye_mp05", "water2", true, true, false, true},
   {"040_water_ast_ne_mp05", "water2", true, true, false, true},
   {"041_water_adt_ye_p10", "water2", false, true, false, true},
   {"042_water_adt_ne_p10", "water2", false, true, false, true},
   {"043_water_adt_ye_p05", "water2", false, true, false, true},
   {"044_water_adt_ne_p05", "water2", false, true, false, true},
   {"045_water_adt_ye_p00", "water2", false, true, false, true},
   {"046_water_adt_ne_p00", "water2", false, true, false, true},
   {"047_water_adt_ye_mp05", "water2", true, true, false, true},
   {"048_water_adt_ne_mp05", "water2", true, true, false, true},
   {"049_water_qnt_ast_l10", "water2", true, true, true, true},
   {"050_water_qnt_ast_l05", "water2", true, true, true, true},
   {"051_water_qnt_ast_l00", "water2", true, true, true, true},
   {"052_water_qnt_adt_l10", "water2", true, true, false, true},
   {"053_water_qnt_adt_l05", "water2", true, true, false, true},
   {"054_water_qnt_adt_l00", "water2", true, true, false, true},
   {"055_water_exp_ast_l10", "water2", true, true, true, true},
   {"056_water_exp_ast_l05", "water2", true, true, true, true},
   {"057_water_exp_ast_l00", "water2", true, true, true, true},
   {"058_water_exp_adt_l10", "water2", true, true, false, true},
   {"059_water_exp_adt_l05", "water2", true, true, false, true},
   {"060_water_exp_adt_l00", "water2", true, true, false, true},
   {"061_water_inv_ast_l10", "water2", true, true, true, true},
   {"062_water_inv_ast_l05", "water2", true, true, true, true},
   {"063_water_inv_ast_l00", "water2", true, true, true, true},
   {"064_water_inv_adt_l10", "water2", true, true, false, true},
   {"065_water_inv_adt_l05", "water2", true, true, false, true},
   {"066_water_inv_adt_l00", "water2", true, true, false, true},
   {"067_water_exf_ast_m10", "water2", true, false, false, true},
   {"068_water_exf_ast_m05", "water2", true, false, false, true},
   {"069_water_exf_ast_m00", "water2", true, false, false, true},
   {"070_water_exf_adt_p10", "water2", false, true, false, true},
   {"071_water_exf_adt_p05", "water2", false, true, false, true},
   {"072_water_exf_adt_p00", "water2", false, true, false, true},
   {"073_water_exf_adt_mp05", "water2", true, true, false, true},
   {"074_water_adt_ye_l10", "water2", true, true, false, true},
   {"075_water_adt_ne_l10", "water2", true, true, false, true},
   {"076_water_adt_ye_l05", "water2", true, true, false, true},
   {"077_water_adt_ne_l05", "water2", true, true, false, true},
   {"078_water_adt_ye_l00", "water2", true, true, false, true},
   {"079_water_adt_ne_l00", "water2", true, true, false, true},
   {"080_water_qnt_adt_l10", "water2", true, true, false, true},
   {"081_water_qnt_adt_l00", "water2", true, true, false, true},
   {"082_water_rels_ye_l100", "water2", true, true, true, true},
   {"083_water_rels_ye_l085", "water2", true, true, true, true},
   {"084_water_rels_ye_l070", "water2", true, true, true, true},
   {"085_water_rels_ye_l050", "water2", true, true, true, true},
   {"086_water_rels_ye_l030", "water2", true, true, true, true},
   {"087_water_rels_ye_l015", "water2", true, true, true, true, nullptr, Tols().grad(1.0e-3, 5.0e-4)},
   {"088_water_rels_ye_l000", "water2", true, true, true, true},
   {"089_water_lmda_ast_l05", "water2", true, true, true, false},
   {"090_water_lmda_ast_e05", "water2", true, true, true, false},
   {"091_water_lmda_ast_l10", "water2", true, true, true, false},
   {"092_water_lmda_ast_none", "water2", true, true, true, false},
   {"093_water_lmda_qnt_l05", "water2", true, true, true, false},
   {"094_water_lmda_vexp_l05", "water2", true, true, true, false},
   {"095_water_lmda_mp05", "water2", true, true, true, false},
   {"096_water_lmda_mp05_expl", "water2", true, true, true, false},
   {"097_water_lmda_e_l06", "water2", true, true, true, false},
   {"098_water_lmda_p_l06", "water2", true, true, true, false},
   {"099_water_lmda_v_l06", "water2", true, true, true, false},
   {"100_water_lmda_ep_l06", "water2", true, true, true, false},
   {"101_water_lmda_ev_l06", "water2", true, true, true, false},
   {"102_water_lmda_pv_l06", "water2", true, true, true, false},
   {"103_water_lmda_epv_l06", "water2", true, true, true, false},
   {"104_water_dlmda_e_l06", "water2", true, true, true, true},
   {"105_water_dlmda_p_l06", "water2", true, true, true, true},
   {"106_water_dlmda_v_l06", "water2", true, true, true, true},
   {"107_water_dlmda_ep_l06", "water2", true, true, true, true},
   {"108_water_dlmda_ev_l06", "water2", true, true, true, true},
   {"109_water_dlmda_pv_l06", "water2", true, true, true, true},
   {"110_water_dlmda_epv_l06", "water2", true, true, true, true},
   {"111_water_rels_ye_vdwm_l030", "water2", true, true, true, true},
   {"112_water_rels_ye_lig1_l070", "water2", true, true, true, true},
   {"113_water_lmda_ast_epin_l05", "water2", true, true, true, true},
   {"114_water_lmda_ast_vpin_l05", "water2", true, true, true, true},
   {"115_water_lmda_adt_vpin_l06", "water2", true, true, true, true},
   {"116_water_rels_ye_vdwm_exp_l050", "water2", true, true, true, true},
   {"117_water_rels_ye_lig2_exp_l030", "water2", true, true, true, true},
   {"118_water_rels_ye_lig1_inv_l070", "water2", true, true, true, true},
   {"119_water_rels_ye_vdwm_vx3_l050", "water2", true, true, true, true},
   {"120_water_rels_ye_lig1_ex3_l085", "water2", true, true, true, true},
   {"121_water_rels_ye_lig2_ix2_l015", "water2", true, true, true, true},
   {"122_water_ast_v05_annihilate", "water2", true, true, true, true},
   {"123_water_exf_adt_l10", "water2", true, true, false, true},
   {"124_water_exf_adt_l05", "water2", true, true, false, true},
   {"125_water_exf_adt_l00", "water2", true, true, false, true},
   {"126_water_ast_ye_l10", "water2", true, true, true, true},
   {"127_water_ast_ne_l10", "water2", true, true, true, true},
   {"128_water_ast_ye_l05", "water2", true, true, true, true},
   {"129_water_ast_ne_l05", "water2", true, true, true, true},
   {"130_water_ast_ye_l00", "water2", true, true, true, true},
   {"131_water_ast_ne_l00", "water2", true, true, true, true},
   {"132_water_exf_ast_l10", "water2", true, true, true, true},
   {"133_water_exf_ast_l05", "water2", true, true, true, true},
   {"134_water_exf_ast_l00", "water2", true, true, true, true},
   {"135_water_apm_ast_vpin_l05", "water2", true, true, true, true},
   {"136_water_apm_ast_epin_l00", "water2", true, true, true, true},
   {"137_water_apm_ast_epin_l05", "water2", true, true, true, true},
   {"138_water_apm_ast_epin_l10", "water2", true, true, true, true},
   {"139_water_vsoft_l10", "water2", true, true, true, true},
   {"140_water_vsoft_l05", "water2", true, true, true, true},
   {"141_water_vsoft_l00", "water2", true, true, true, true},
   {"142_water_rels_st_l085", "water2", true, true, true, true},
   {"143_water_rels_st_lig2_exp_l030", "water2", true, true, true, true},
   {"144_water_rels_nolmda", "water2", true, true, true, true, "082_water_rels_ye_l100"},
   {"145_trpcage_chiral_m05", "trpcage", false, false, false, false},
   {"146_g3_ast_ye_l10", "g3", true, true, true, true},
   {"147_g3_ast_ye_l05", "g3", true, true, true, true},
   {"148_g3_ast_annih_l05", "g3", true, true, true, true},
   {"149_g3_ast_nobox_l05", "g3", true, true, true, true},
   {"150_water_ast_vcorr_annih_l05", "water2", true, true, true, true, nullptr, Tols().virial(1.0e-2, 1.0e-2)},
   {"151_water_ast_mono_l05", "water2", true, true, true, true},
   {"152_water_ast_tric_l05", "water2", true, true, true, true},
   {"153_water_rels_ye_vdwm_lig2t_l040", "water2", true, true, true, true},
   {"154_water_ast_ne_mcut_l05", "water2", true, true, true, true},
   {"155_g3_rels_lig1_l085", "g3", true, true, true, true, nullptr, Tols().grad(2.0e-3, 5.0e-4)},
   {"156_g3_rels_lig2_l015", "g3", true, true, true, true, nullptr, Tols().lmda2(2.0e-3, 1.0e-4)},
   {"157_water_rels_ne_lig2_l015", "water2", true, true, true, true},
   {"158_frames_ast_ye_l05", "frames", true, true, true, true},
   {"159_frames_ast_nobox_l05", "frames", true, true, true, true},
   {"160_chig_ast_nobox_l05", "chig", true, true, true, true, nullptr, Tols().grad(1.0e-3, 5.0e-4)},
   {"161_chigm_ast_nobox_l05", "chigm", true, true, true, true, nullptr, Tols().grad(1.0e-3, 5.0e-4)},
   {"162_chig_ast_ye_l05", "chig", true, true, true, true, nullptr, Tols().grad(1.0e-3, 5.0e-4)},
   {"163_chigm_ast_ye_l05", "chigm", true, true, true, true, nullptr, Tols().grad(1.0e-3, 5.0e-4)},
   {"164_water_ast_ne_mcut_d1_l05", "water2", true, true, true, true},
   {"165_g3_ast_d1_l05", "g3", true, true, true, true},
   {"166_g3_ast_d1_l00", "g3", true, true, true, true},
   {"167_g3_rels_lig1_d1_l085", "g3", true, true, true, true},
   {"168_g3_rels_lig2_d1_l015", "g3", true, true, true, true},
   {"169_water_adt_d1_x2_l06", "water2", false, true, false, true},
   {"170_water_ast_vcorr_annih_d1_l05", "water2", true, true, true, true},
   {"171_water_rels_ye_vdwm_d1_l040", "water2", true, true, true, true},
   {"172_water_vsoft_n1_d1_l00", "water2", true, true, true, true},
   {"173_water_vsoft_n1_d1_l005", "water2", true, true, true, true},
   {"174_water_rels_vdwm_n1_d1_l10", "water2", true, true, true, true},
   {"175_water_vsoft_n15_ti_l00", "water2", true, true, true, true, "172_water_vsoft_n1_d1_l00"},
   {"176_water_rels_st_ne_l085", "water2", true, true, true, true},
   {"177_water_rels_ne_l085", "water2", true, true, true, true},
   {"178_water_adt_d1_ne_l06", "water2", false, true, false, true},
   {"179_ionwat_ast_l05", "ionwat", true, true, true, true},
};

// The fixture of a given name. Cases look their fixture up by name so that
// retiring a fixture does not shift every case after it.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no mutate fixture named " << name);
   return kFixtures[0];
}

// The directory holding a base system's coordinates: the SAMPL8 guest 3 and
// trp-cage have directories of their own, and the other systems sit with the
// mutation fixtures.
std::string systemDir(const std::string& base)
{
   if (base == "g3")
      return TINKER9_DIRSTR "/test/file/g3/";
   if (base == "trpcage")
      return TINKER9_DIRSTR "/test/file/trpcage/";
   return TINKER9_DIRSTR "/test/file/mutate/";
}

// Copies the parameter files a base system loads into the working directory.
// The water fixtures share water03; the SAMPL8 guest 3 carries its own force
// field, plus the artificial vdw14 values fixture 148 loads as a second file.
// The frames dimer and the ion cluster load amoeba09, chignolin and its mirror
// amoebabio09, and trp-cage amoebapro13.
std::vector<std::unique_ptr<TestFile>> copyParams(const std::string& base)
{
   std::vector<std::unique_ptr<TestFile>> files;
   if (base == "g3") {
      std::string dir = systemDir(base);
      files.emplace_back(new TestFile(dir + "g3.prm"));
      files.emplace_back(new TestFile(dir + "g3_vdw14.prm"));
   } else if (base == "frames" or base == "ionwat") {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoeba09.prm"));
   } else if (base == "chig" or base == "chigm") {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoebabio09.prm"));
   } else if (base == "trpcage") {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_291a85c1/amoebapro13.prm"));
   } else {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm"));
   }
   return files;
}

// The extra key lines of a fixture. The OpenACC build names its own package, so
// that a run there exercises the OpenACC kernels; runFixture checks that it did,
// since a GPU_PACKAGE environment variable overrides the key.
std::string platformKey(const char* keyextra)
{
#if TINKER_GPULANG_OPENACC
   return std::string(keyextra) + "\ngpu-package openacc\n";
#else
   return keyextra;
#endif
}

// Copies a fixture's coordinates, its key file with \c keyextra appended, and
// its parameters into the working directory, and begins a session on them with
// \c rc as rc_flag. The session is declared after the files, so it ends before
// they are removed, and the lambda flags are reset after it ends. With \c init
// false the caller runs session.init() itself, after adjusting the Fortran side.
struct Setup
{
   TestLmdaFlagReset lmdaReset;
   std::string xyz, key;
   const char* argv[4];
   TestFile fxyz, fkey;
   std::vector<std::unique_ptr<TestFile>> fprm;
   TestSession session;

   Setup(const Fixture& fx, int rc, const char* keyextra = "", bool init = true)
      : xyz(std::string(fx.base) + ".xyz")
      , key(std::string(fx.name) + ".key")
      , argv{"dummy", xyz.c_str(), "-k", key.c_str()}
      , fxyz(systemDir(fx.base) + xyz, xyz)
      , fkey(TINKER9_DIRSTR "/test/file/mutate/" + key, key, platformKey(keyextra))
      , fprm(copyParams(fx.base))
      , session(4, argv)
   {
      rc_flag = rc;
      if (init)
         session.init();
   }
};

// How a run should treat the fused multipole/polarization kernel. emplar cannot
// report interaction counts, so any evaluation that asks for them routes around
// it -- which is why an ordinary run never exercises it at all.
//
// Dropping counts also drops analysis, which is how dynamics runs: the terms then
// keep no lambda derivative buffers of their own and add straight into the
// shared dedl_buf and d2edl2_buf (LmdaBuffer).
enum class Fuse
{
   Off,     ///< Ordinary run: counts are requested, so emplar is out of reach.
   Require, ///< Drop counts, and require that emplar took over.
   Forbid,  ///< Drop counts, and require that it still did not.
   Auto     ///< Drop counts, as dynamics does, and take whichever kernel is chosen.
};

enum class LmdaMode
{
   Default,
   /// Appends LAMBDA-MODE TI to the key file. The fixture keeps LAMBDA-DERIV, so
   /// the lambda-driven kernels still receive v9 for v1 and v10 for v4.
   ThermIntg,
};

void runFixture(const Fixture& fx, Fuse fuse = Fuse::Off, LmdaMode lmdaMode = LmdaMode::Default)
{
   std::string refpath = std::string(TINKER9_DIRSTR "/test/ref/mutate/") + (fx.ref ? fx.ref : fx.name) + ".txt";

   const double eps_e = Tols::pick(fx.tol.e, 1.0e-3, 1.0e-4);
   const double eps_g = Tols::pick(fx.tol.g, 1.0e-3, 1.0e-4);
   const double eps_v = Tols::pick(fx.tol.v, 2.0e-3, 1.0e-3);
   // dV/dL is a difference of two endpoint virials of comparable size, so it
   // loses the leading digits the plain virial keeps, and the references print
   // it to three decimals. testlmda.cpp uses the same allowance.
   const double eps_dv = std::max(eps_v, testGetEps(1.0e-2, 2.0e-3));
   const double eps_l = Tols::pick(fx.tol.l, 1.0e-3, 1.0e-4);
   const double eps_l2 = Tols::pick(fx.tol.l2, 1.0e-3, 1.0e-4);

   int rc = calc::xyz | calc::mass | calc::vmask;
   if (fuse != Fuse::Off)
      rc &= ~calc::analyz;

   // TI owns the main lambda and starts at the first schedule window. The TI
   // fixtures all reference lambda 0.5, so keep that operating point instead of
   // accepting TI's default first window at lambda 1.
   const char* keyextra = lmdaMode == LmdaMode::ThermIntg ? "\nlambda-mode ti\nti-window 0.5\n" : "";
   // These key files enable the lambda-derivative machinery through the
   // "lambda-deriv" keyword, so the Fortran-side use_dlmda needs no nudging here.
   Setup s(fx, rc, keyextra);
#if TINKER_GPULANG_OPENACC
   REQUIRE(pltfm_config & Platform::ACC);
#endif

   // The fixtures carry LAMBDA-DERIV, which asks for every derivative channel
   // whatever the sampling mode; without it TI, META and ABF would ask for the
   // first energy derivative alone (mutate.f: use_d2lmda). Single topology
   // polarization collapses the dispatch the same way: past dE/dL the chain rule
   // has no single-topology form, so d2E/dL2, dF/dL and dV/dL stay at zero and
   // the kernels are never asked for them. Verify the dispatch contract here as
   // well as checking the resulting quantities below.
   const bool tiMode = lmdaMode == LmdaMode::ThermIntg;
   const bool reducedLmda = not use_d2lmda;
   if (use_dlmda) {
      REQUIRE(lmdaDerivVers(calc::v1, use_dlmda) == (reducedLmda ? calc::v7 : calc::v9));
      REQUIRE(lmdaDerivVers(calc::v4, use_dlmda) == (reducedLmda ? calc::v8 : calc::v10));
   }
   if (tiMode) {
      REQUIRE(use_dlmda);
      REQUIRE(use_ti);
      REQUIRE(use_mainlmda);
      // The absolute dual topology fixture runs without van der Waals.
      REQUIRE(use_vdlmda == fx.checkv);
      COMPARE_REALS(lambda, 0.5, eps_l);
   }

   if (fuse == Fuse::Require)
      REQUIRE(useEmplar());
   else if (fuse == Fuse::Forbid)
      REQUIRE_FALSE(useEmplar());

   // Without analysis every term must add into the shared lambda derivative
   // buffers rather than keep its own, or the checks below would not be testing
   // the dynamics layout at all.
   if (fuse != Fuse::Off) {
      REQUIRE_FALSE(rc_flag & calc::analyz);
      auto shared = [](EnergyBuffer term, EnergyBuffer total) { return term == nullptr or term == total; };
      REQUIRE(shared(demdl_buf, dedl_buf));
      REQUIRE(shared(depdl_buf, dedl_buf));
      REQUIRE(shared(devdl_buf, dedl_buf));
      REQUIRE(shared(d2emdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2epdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2evdl2_buf, d2edl2_buf));
      if (fx.dolmda)
         REQUIRE(dedl_buf != nullptr);
   }

   TestReference ref(refpath);
   double ref_e = ref.getEnergy();
   auto ref_v = ref.getVirial();
   auto ref_g = ref.getGradient();

   // The lambda-derivative sections are absent from the non-dolmda references,
   // in which case every field reads back as zero and goes unused.
   const TestLmdaReference& lr = ref.getLmda();
   if (fx.dolmda)
      REQUIRE((int)lr.lgrad.size() >= n);

   // The scalar lambda derivatives vs reference. The per-term split is an
   // analysis feature: only there does a term keep an energy derivative buffer
   // of its own, and everywhere else the terms add into the shared one, which is
   // reduced straight into the total. So the breakdown is checked under analysis
   // and the total is checked always. Polarization is the exception either way:
   // its derivative arrives in sub-lambda units, so it stays private whenever it
   // is driven, and it is folded into the total already scaled.
   const bool splitTerms = rc_flag & calc::analyz;
   auto checkLmdaFirstScalars = [&]() {
      COMPARE_REALS(dedl, lr.dedl[0], eps_l);
      if (not splitTerms)
         return;
      COMPARE_REALS(devdl, lr.dedl[1], eps_l);
      if (fuse == Fuse::Require) {
         COMPARE_REALS(demdl + depdl, lr.dedl[2] + lr.dedl[3], eps_l);
      } else {
         COMPARE_REALS(demdl, lr.dedl[2], eps_l);
         COMPARE_REALS(depdl, lr.dedl[3], eps_l);
      }
   };

   auto checkLmdaSecondScalars = [&]() {
      COMPARE_REALS(d2edl2, lr.d2edl2[0], eps_l2);
      if (not splitTerms)
         return;
      COMPARE_REALS(d2evdl2, lr.d2edl2[1], eps_l2);
      if (fuse == Fuse::Require) {
         COMPARE_REALS(d2emdl2 + d2epdl2, lr.d2edl2[2] + lr.d2edl2[3], eps_l2);
      } else {
         COMPARE_REALS(d2emdl2, lr.d2edl2[2], eps_l2);
         COMPARE_REALS(d2epdl2, lr.d2edl2[3], eps_l2);
      }
   };

   // Per-atom lambda gradient (dfdl*) vs reference. Valid whenever calc::grad
   // is requested, since the terms accumulate dfdl* whenever they build a gradient.
   auto checkLmdaGrad = [&]() {
      std::vector<double> lx(n), ly(n), lz(n);
      copyGradient(calc::grad, lx.data(), ly.data(), lz.data(), dfdlx, dfdly, dfdlz);
      for (int i = 0; i < n; ++i) {
         COMPARE_REALS(lx[i], lr.lgrad[i][0], eps_g);
         COMPARE_REALS(ly[i], lr.lgrad[i][1], eps_g);
         COMPARE_REALS(lz[i], lr.lgrad[i][2], eps_g);
      }
   };

   // v0
   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   // v1
   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   COMPARE_VIR9(vir, ref_v, eps_v);

   if (fx.dolmda) {
      checkLmdaFirstScalars();
      if (not reducedLmda) {
         checkLmdaSecondScalars();
         checkLmdaGrad();
         COMPARE_VIR9(dvirdl, lr.dvdl, eps_dv);
      }
   }

   // v3 -- the count buffers are allocated only under calc::analyz.
   if (fuse == Fuse::Off) {
      energy(calc::v3);
      COMPARE_REALS(esum, ref_e, eps_e);
      double eng;
      int cnt;
      if (fx.checkm) {
         ref.getEnergyCountByName("Atomic Multipoles", eng, cnt);
         COMPARE_COUNT(nem, cnt);
         COMPARE_ENERGY(em, eng, eps_e);
      }
      if (fx.checkp) {
         ref.getEnergyCountByName("Polarization", eng, cnt);
         COMPARE_COUNT(nep, cnt);
         COMPARE_ENERGY(ep, eng, eps_e);
      }
      if (fx.checkv) {
         ref.getEnergyCountByName("Van der Waals", eng, cnt);
         COMPARE_COUNT(nev, cnt);
         COMPARE_ENERGY(ev, eng, eps_e);
      }
   }

   // v4
   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   if (fx.dolmda) {
      checkLmdaFirstScalars();
      if (not reducedLmda) {
         checkLmdaSecondScalars();
         checkLmdaGrad();
      }
   }

   // level 5 -- gradient only (no energy, no virial)
   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   // level 6 -- gradient + virial (no energy)
   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   COMPARE_VIR9(vir, ref_v, eps_v);

   s.session.end();
}

#if TINKER_GPULANG_CUDA
// Single topology electrostatics that the fused kernel still has to decline:
// without single topology polarization alongside it, the dispatch is not the
// reduced one emplarast is built for. Dropping the counts is what makes this
// check meaningful: with them, emplar is refused for every fixture and the
// assertion would pass for the wrong reason.
void runNoEmplarFixture(const Fixture& fx)
{
   runFixture(fx);
   runFixture(fx, Fuse::Forbid);
   REQUIRE(use_emast);
   REQUIRE_FALSE(use_epast);
}

// Single topology on both terms, run twice: once asking for interaction counts,
// which routes around emplar to the separate empole and epolar kernels, and once
// without them, where emplarast fuses the two. Both are checked against the same
// reference, so the fused path has to agree with the split one and with Tinker.
// use_epast also reduces the derivative dispatch to dE/dL alone, so the
// references carry zeros for d2E/dL2, dF/dL and dV/dL.
void runEmplarAstFixture(const Fixture& fx)
{
   runFixture(fx);
   runFixture(fx, Fuse::Require);
   REQUIRE(use_emast);
   REQUIRE(use_epast);
}

// Checks which dual topology endpoints the run had to build. dtNeed() decides
// this per term from the interpolation weight and the chain rule, so it is
// asserted against the sub-lambda state the fixture left behind.
void runLegSkipFixture(const Fixture& fx, bool expect0, bool expect1)
{
   runFixture(fx);

   double w, dw, d2w;
   bool need0, need1;

   dtWeightNeed(plam, epdtexp, dpldlmda, d2pldlmda2, w, dw, d2w, need0, need1);
   REQUIRE(need0 == expect0);
   REQUIRE(need1 == expect1);
}

// Sum of |dF/dL| over every atom, zero if the force lambda derivative has no
// storage.
double sumAbsDfdl()
{
   if (not dfdlx)
      return 0;
   std::vector<double> lx(n), ly(n), lz(n);
   copyGradient(calc::grad, lx.data(), ly.data(), lz.data(), dfdlx, dfdly, dfdlz);
   double s = 0;
   for (int i = 0; i < n; ++i)
      s += std::fabs(lx[i]) + std::fabs(ly[i]) + std::fabs(lz[i]);
   return s;
}

// Sum of |dV/dL| over the nine tensor components.
double sumAbsDvirdl()
{
   double s = 0;
   for (int i = 0; i < 9; ++i)
      s += std::fabs(dvirdl[i]);
   return s;
}

// Runs a fixture whose multipole second lambda derivative is nonzero with and
// without the second, force and virial lambda derivatives (test_mutate_gate).
// Turning them off must leave the energy, the gradient, the virial and the
// first lambda derivatives unchanged, and clear the second, force and virial
// lambda derivatives that the full run left behind.
void runGateFixture(const Fixture& fx)
{
   const double eps = testGetEps(1.0e-4, 1.0e-8);

   Setup s(fx, calc::xyz | calc::mass | calc::vmask);

   // Full lambda derivatives, as the LAMBDA-DERIV keyword requests.
   REQUIRE(use_d2lmda);
   energy(calc::v1);
   const double e1 = esum, dedl1 = dedl, demdl1 = demdl;
   std::vector<double> gx1(n), gy1(n), gz1(n);
   copyGradient(calc::grad, gx1.data(), gy1.data(), gz1.data());
   double vir1[9];
   for (int i = 0; i < 9; ++i)
      vir1[i] = vir[i];
   REQUIRE(d2emdl2 != 0);
   REQUIRE(dfdlx);
   REQUIRE(sumAbsDfdl() != 0);
   REQUIRE(sumAbsDvirdl() != 0);

   // Only the first lambda derivative, as TI, META and ABF request.
   use_d2lmda = false;
   REQUIRE(lmdaDerivVers(calc::v1, true) == calc::v7);
   energy(calc::v1);
   COMPARE_REALS(esum, e1, eps);
   COMPARE_REALS(dedl, dedl1, eps);
   COMPARE_REALS(demdl, demdl1, eps);
   std::vector<double> gx(n), gy(n), gz(n);
   copyGradient(calc::grad, gx.data(), gy.data(), gz.data());
   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(gx[i], gx1[i], eps);
      COMPARE_REALS(gy[i], gy1[i], eps);
      COMPARE_REALS(gz[i], gz1[i], eps);
   }
   for (int i = 0; i < 9; ++i)
      COMPARE_REALS(vir[i], vir1[i], eps);
   REQUIRE(d2emdl2 == 0);
   REQUIRE(d2edl2 == 0);
   REQUIRE(sumAbsDfdl() == 0);
   REQUIRE(sumAbsDvirdl() == 0);

   s.session.end();
}

// Runs a soft core exponent below two under TI at lambda 0, the endpoint where
// the second lambda derivative factor v^(n-2) diverges (test_mutate_scexp).
// TI asks for dE/dL alone, so the run must stay finite. At a van der Waals
// lambda of zero the coupled terms vanish whatever the exponent, so the energy,
// gradient and virial are those of the borrowed n = 1 reference, while dE/dL,
// which carries a factor of lambda^(n-1), drops to zero.
void runScexpTiFixture(const Fixture& fx)
{
   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, 1.0e-4);
   const double eps_v = testGetEps(2.0e-3, 1.0e-3);

   // TI would start at its default first window, lambda 1.
   Setup s(fx, calc::xyz | calc::mass | calc::vmask, "\nti-window 0.0\n");
   REQUIRE(use_ti);
   REQUIRE_FALSE(use_d2lmda);
   REQUIRE(lambda == 0);

   energy(calc::v1);
   REQUIRE(std::isfinite(esum));
   REQUIRE(std::isfinite(dedl));
   std::vector<double> gx(n), gy(n), gz(n);
   copyGradient(calc::grad, gx.data(), gy.data(), gz.data());
   for (int i = 0; i < n; ++i)
      REQUIRE((std::isfinite(gx[i]) and std::isfinite(gy[i]) and std::isfinite(gz[i])));
   for (int i = 0; i < 9; ++i)
      REQUIRE(std::isfinite(vir[i]));

   TestReference ref(std::string(TINKER9_DIRSTR "/test/ref/mutate/") + fx.ref + ".txt");
   auto ref_g = ref.getGradient();
   auto ref_v = ref.getVirial();
   COMPARE_REALS(esum, ref.getEnergy(), eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   COMPARE_VIR9(vir, ref_v, eps_v);

   REQUIRE(std::fabs(dedl) <= 1.0e-8);
   REQUIRE(d2edl2 == 0);
   REQUIRE(sumAbsDfdl() == 0);
   REQUIRE(sumAbsDvirdl() == 0);

   s.session.end();
}

// One multipole or polarization result of runFlatFixture.
struct FlatTerm
{
   double e;
   std::array<double, 9> v;
   std::vector<double> gx, gy, gz;
};

FlatTerm flatTerm(double e, const virial_prec* v, const grad_prec* gx, const grad_prec* gy, const grad_prec* gz)
{
   FlatTerm t;
   t.e = e;
   for (int i = 0; i < 9; ++i)
      t.v[i] = v[i];
   t.gx.resize(n);
   t.gy.resize(n);
   t.gz.resize(n);
   copyGradient(calc::grad, t.gx.data(), t.gy.data(), t.gz.data(), gx, gy, gz);
   return t;
}

void compareFlatTerm(const FlatTerm& got, const FlatTerm& ref, double eps)
{
   COMPARE_REALS(got.e, ref.e, eps * std::max(1.0, std::fabs(ref.e)));
   for (int i = 0; i < 9; ++i)
      COMPARE_REALS(got.v[i], ref.v[i], eps * std::max(1.0, std::fabs(ref.v[i])));
   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(got.gx[i], ref.gx[i], eps);
      COMPARE_REALS(got.gy[i], ref.gy[i], eps);
      COMPARE_REALS(got.gz[i], ref.gz[i], eps);
   }
}

// Moves a staged ligand 1 charging leg below its 0.7 to 1.0 electrostatic
// window, where the quintic map is flat and energy() takes the plain versions
// (test_mutate_flat). The lambda derivatives must be exact zeros there, giving
// the flat maps a slope must leave the energy, gradient and virial alone, and
// moving back inside the window must restore the multipole lambda derivative.
// Analysis runs empole and epolar apart, so each is compared on its own; fused
// runs emplarAst, compared through the electrostatic accumulators it adds into.
void runFlatFixture(const Fixture& fx, bool fused)
{
   // Grid spreading adds floats atomically, so a rebuild is not bit identical.
   const double eps = testGetEps(1.0e-4, 1.0e-6);

   int rc = calc::xyz | calc::mass | calc::vmask;
   if (fused)
      rc &= ~calc::analyz;
   Setup s(fx, rc);

   REQUIRE(use_mainlmda);
   REQUIRE(use_edlmda);
   REQUIRE(use_pdlmda);
   REQUIRE(useEmplar() == fused);

   // Below the window the chain rule is flat and every lambda derivative is an
   // exact zero.
   lambda = 0.5;
   energy(calc::v1);
   REQUIRE_FALSE(edlmdaActive());
   REQUIRE_FALSE(pdlmdaActive());
   REQUIRE(deldlmda == 0);
   REQUIRE(dpldlmda == 0);
   REQUIRE(dedl == 0);
   REQUIRE(d2edl2 == 0);
   if (not fused) {
      REQUIRE(demdl == 0);
      REQUIRE(depdl == 0);
   }
   REQUIRE(sumAbsDfdl() == 0);
   REQUIRE(sumAbsDvirdl() == 0);

   // A slope only scales the lambda derivative channels, so it must leave the
   // energy, gradient and virial as the plain versions left them, which is what
   // lets energy() skip the lambda derivative ones on a flat map. Polarization
   // may reuse the reciprocal multipole potential and cmp the multipole call
   // just before it left, so each polarization call follows one, and the mixed
   // pairs cover maps whose windows differ.
   auto flatPair = [&](int vers, bool me, bool pe, FlatTerm& em_out, FlatTerm& ep_out) {
      deldlmda = me ? 1 : 0;
      dpldlmda = pe ? 1 : 0;
      REQUIRE(edlmdaActive() == me);
      REQUIRE(pdlmdaActive() == pe);
      zeroEGV(vers);
      // Without calc::virial nothing clears the virial, so compare zeros instead.
      const bool do_v = vers & calc::virial;
      const virial_prec zero9[9] = {0};
      if (fused) {
         emplarAst(vers);
         virial_prec v[9] = {0};
         if (do_v) {
            virialReduce(v, vir_buf_elec);
            for (int i = 0; i < 9; ++i)
               v[i] += virial_elec[i];
         }
         em_out = flatTerm(energyReduce(eng_buf_elec), v, gx_elec, gy_elec, gz_elec);
      } else {
         empole(vers);
         if (use_epdt)
            epolar_dt(vers);
         else
            epolar(vers);
         em_out = flatTerm(energy_em, do_v ? virial_em : zero9, demx, demy, demz);
         ep_out = flatTerm(energy_ep, do_v ? virial_ep : zero9, depx, depy, depz);
      }
   };

   // Both the energy-gradient-virial and the energy-gradient versions, each of
   // which has a plain kernel version of its own.
   for (int vers : {calc::v1, calc::v4}) {
      CAPTURE(vers);
      FlatTerm em0, ep0, em1, ep1;
      flatPair(vers, false, false, em0, ep0);
      for (auto mp : {std::make_pair(true, true), std::make_pair(true, false), std::make_pair(false, true)}) {
         CAPTURE(mp.first, mp.second);
         flatPair(vers, mp.first, mp.second, em1, ep1);
         compareFlatTerm(em1, em0, eps);
         if (not fused) {
            compareFlatTerm(ep1, ep0, eps);
            // the slope did send each term down its lambda derivative version
            REQUIRE((demdl != 0) == mp.first);
            REQUIRE((depdl != 0) == mp.second);
         }
      }
   }
   deldlmda = 0;
   dpldlmda = 0;

   // Back inside the window the multipole lambda derivative returns.
   lambda = 0.85;
   energy(calc::v1);
   REQUIRE(edlmdaActive());
   REQUIRE(dedl != 0);
   if (not fused)
      REQUIRE(demdl != 0);

   s.session.end();
}

// Loads a fixture, with an optional keyword appended to its key file, and
// checks how the derivative keywords choose the polarization topology: the
// first derivative alone keeps single topology, while LAMBDA-DERIV2 needs the
// second derivatives and with them dual topology (test_mutate.f).
void runFlagsFixture(const Fixture& fx, const char* keyextra, bool d2, bool epdt, bool rel)
{
   Setup s(fx, calc::xyz | calc::mass | calc::vmask, keyextra);

   REQUIRE(use_dlmda);
   REQUIRE(use_d2lmda == d2);
   REQUIRE(use_epdt == epdt);
   REQUIRE((dlmda::use_prst != 0) == not epdt);
   REQUIRE(use_rel == rel);

   s.session.end();
}

// Mutates the first residues of trp-cage, whose alpha carbons carry chiral
// multipole frames, with the multipole term alone at a fixed electrostatic
// lambda (test_mutate.f:test_mutate_chiral). The shipped alpha carbon
// multipoles have no y components, so a y dipole is seeded at every chiral site
// to make the inversion visible. Mirroring the coordinates inverts every chiral
// frame; chkpole must flip poleorig along with pole, so that a later rescale of
// pole from poleorig keeps the inversion.
//
// Tinker allocates poleorig for any mutation, while tinker9 keeps it on the
// device only for a lambda derivative, so one is asked for here. A derivative
// needs a main lambda, and a main lambda needs a map; it drives van der Waals
// alone, which this fixture leaves off, so elambda stays pinned by its keyword.
void runChiralFixture(const Fixture& fx)
{
   const double eps_e = testGetEps(1.0e-3, 1.0e-8);
   const double eps_p = testGetEps(1.0e-6, 1.0e-12);

   Setup s(fx, calc::xyz | calc::mass | calc::energy, "\nlambda-deriv\nlambda 0.5\nvdw-lmda-map exp\n", false);

   // tinker9 sets n in initialize(); the Fortran side is already set up.
   std::vector<int> chiral;
   for (int i = 0; i < atoms::n; ++i) {
      if (std::strncmp(mpole::polaxe[i], "Z-then-X", 8) == 0 and mpole::yaxis[i] != 0) {
         dlmda::poleorig[mpole::maxpole * i + 2] = 0.1;
         chiral.push_back(i);
      }
   }
   REQUIRE(chiral.size() > 0);
   tinker_f_altelec();

   s.session.init();
   REQUIRE(poleorig != nullptr);
   REQUIRE_FALSE(use_emast);

   auto readY = [&](real (*arr)[MPL_TOTAL]) {
      std::vector<std::array<real, MPL_TOTAL>> buf(n);
      darray::copyout(g::q0, n, reinterpret_cast<real(*)[MPL_TOTAL]>(buf.data()), arr);
      waitFor(g::q0);
      std::vector<double> y;
      for (int i : chiral)
         y.push_back(buf[i][MPL_PME_Y]);
      return y;
   };

   energy(calc::v0);
   const double e0 = esum;
   const auto pole0 = readY(pole);
   const auto orig0 = readY(poleorig);

   // Mirror the structure, which inverts every chiral frame.
   std::vector<pos_prec> xbuf(n);
   darray::copyout(g::q0, n, xbuf.data(), xpos);
   waitFor(g::q0);
   for (auto& v : xbuf)
      v = -v;
   darray::copyin(g::q0, n, xpos, xbuf.data());
   waitFor(g::q0);
   copyPosToXyz(true);

   energy(calc::v0);
   COMPARE_REALS(esum, e0, eps_e);

   // pole and poleorig flipped together, so pole is still the scaled poleorig
   // and a rescale from poleorig reproduces it.
   const auto pole1 = readY(pole);
   const auto orig1 = readY(poleorig);
   for (size_t k = 0; k < chiral.size(); ++k) {
      const int i = chiral[k];
      const double sc = mutant::mutg[i] != 0 ? elam : 1.0;
      COMPARE_REALS(pole1[k], -pole0[k], eps_p);
      COMPARE_REALS(orig1[k], -orig0[k], eps_p);
      COMPARE_REALS(pole1[k], sc * orig1[k], eps_p);
   }

   s.session.end();
}
#endif
} // namespace

TEST_CASE("MUTATE-001_water_ye_m10", "[ff][mutate][mv]") { runFixture(fx("001_water_ye_m10")); }
TEST_CASE("MUTATE-002_water_ne_m10", "[ff][mutate][mv]") { runFixture(fx("002_water_ne_m10")); }
TEST_CASE("MUTATE-003_water_ye_m05", "[ff][mutate][mv]") { runFixture(fx("003_water_ye_m05")); }
TEST_CASE("MUTATE-004_water_ne_m05", "[ff][mutate][mv]") { runFixture(fx("004_water_ne_m05")); }
TEST_CASE("MUTATE-005_water_ye_m00", "[ff][mutate][mv]") { runFixture(fx("005_water_ye_m00")); }
TEST_CASE("MUTATE-006_water_ne_m00", "[ff][mutate][mv]") { runFixture(fx("006_water_ne_m00")); }
TEST_CASE("MUTATE-007_water_v10", "[ff][mutate][mv]") { runFixture(fx("007_water_v10")); }
TEST_CASE("MUTATE-008_water_v05", "[ff][mutate][mv]") { runFixture(fx("008_water_v05")); }
TEST_CASE("MUTATE-009_water_v00", "[ff][mutate][mv]") { runFixture(fx("009_water_v00")); }
TEST_CASE("MUTATE-010_water_ye_m10", "[ff][mutate][mp]") { runFixture(fx("010_water_ye_m10")); }
TEST_CASE("MUTATE-011_water_ne_m10", "[ff][mutate][mp]") { runFixture(fx("011_water_ne_m10")); }
TEST_CASE("MUTATE-012_water_ye_m05", "[ff][mutate][mp]") { runFixture(fx("012_water_ye_m05")); }
TEST_CASE("MUTATE-013_water_ne_m05", "[ff][mutate][mp]") { runFixture(fx("013_water_ne_m05")); }
TEST_CASE("MUTATE-014_water_ye_m00", "[ff][mutate][mp]") { runFixture(fx("014_water_ye_m00")); }
TEST_CASE("MUTATE-015_water_ne_m00", "[ff][mutate][mp]") { runFixture(fx("015_water_ne_m00")); }
TEST_CASE("MUTATE-016_water_ye_p10", "[ff][mutate][mp]") { runFixture(fx("016_water_ye_p10")); }
TEST_CASE("MUTATE-017_water_ne_p10", "[ff][mutate][mp]") { runFixture(fx("017_water_ne_p10")); }
TEST_CASE("MUTATE-018_water_ye_p05", "[ff][mutate][mp]") { runFixture(fx("018_water_ye_p05")); }
TEST_CASE("MUTATE-019_water_ne_p05", "[ff][mutate][mp]") { runFixture(fx("019_water_ne_p05")); }
TEST_CASE("MUTATE-020_water_ye_p00", "[ff][mutate][mp]") { runFixture(fx("020_water_ye_p00")); }
TEST_CASE("MUTATE-021_water_ne_p00", "[ff][mutate][mp]") { runFixture(fx("021_water_ne_p00")); }
TEST_CASE("MUTATE-022_water_ye_m10p05", "[ff][mutate][mp]") { runFixture(fx("022_water_ye_m10p05")); }
TEST_CASE("MUTATE-023_water_ne_m10p05", "[ff][mutate][mp]") { runFixture(fx("023_water_ne_m10p05")); }
TEST_CASE("MUTATE-024_water_ye_m05p10", "[ff][mutate][mp]") { runFixture(fx("024_water_ye_m05p10")); }
TEST_CASE("MUTATE-025_water_ne_m05p10", "[ff][mutate][mp]") { runFixture(fx("025_water_ne_m05p10")); }
TEST_CASE("MUTATE-026_water_ye_m05p00", "[ff][mutate][mp]") { runFixture(fx("026_water_ye_m05p00")); }
TEST_CASE("MUTATE-027_water_ne_m05p00", "[ff][mutate][mp]") { runFixture(fx("027_water_ne_m05p00")); }
TEST_CASE("MUTATE-028_water_ye_m00p05", "[ff][mutate][mp]") { runFixture(fx("028_water_ye_m00p05")); }
TEST_CASE("MUTATE-029_water_ne_m00p05", "[ff][mutate][mp]") { runFixture(fx("029_water_ne_m00p05")); }

TEST_CASE("MUTATE-089_water_lmda_ast_l05", "[ff][mutate][lmda]") { runFixture(fx("089_water_lmda_ast_l05")); }
TEST_CASE("MUTATE-090_water_lmda_ast_e05", "[ff][mutate][lmda]") { runFixture(fx("090_water_lmda_ast_e05")); }
TEST_CASE("MUTATE-091_water_lmda_ast_l10", "[ff][mutate][lmda]") { runFixture(fx("091_water_lmda_ast_l10")); }
TEST_CASE("MUTATE-092_water_lmda_ast_none", "[ff][mutate][lmda]") { runFixture(fx("092_water_lmda_ast_none")); }
TEST_CASE("MUTATE-093_water_lmda_qnt_l05", "[ff][mutate][lmda]") { runFixture(fx("093_water_lmda_qnt_l05")); }
TEST_CASE("MUTATE-094_water_lmda_vexp_l05", "[ff][mutate][lmda]") { runFixture(fx("094_water_lmda_vexp_l05")); }
TEST_CASE("MUTATE-095_water_lmda_mp05", "[ff][mutate][lmda]") { runFixture(fx("095_water_lmda_mp05")); }
TEST_CASE("MUTATE-096_water_lmda_mp05_expl", "[ff][mutate][lmda]") { runFixture(fx("096_water_lmda_mp05_expl")); }
TEST_CASE("MUTATE-097_water_lmda_e_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("097_water_lmda_e_l06")); }
TEST_CASE("MUTATE-099_water_lmda_v_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("099_water_lmda_v_l06")); }
TEST_CASE("MUTATE-101_water_lmda_ev_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("101_water_lmda_ev_l06")); }

#if TINKER_GPULANG_CUDA
TEST_CASE("MUTATE-030_water_ast_ye_m10", "[ff][mutate][ast]") { runFixture(fx("030_water_ast_ye_m10")); }
TEST_CASE("MUTATE-031_water_ast_ne_m10", "[ff][mutate][ast]") { runFixture(fx("031_water_ast_ne_m10")); }
TEST_CASE("MUTATE-032_water_ast_ye_m05", "[ff][mutate][ast]") { runFixture(fx("032_water_ast_ye_m05")); }
TEST_CASE("MUTATE-033_water_ast_ne_m05", "[ff][mutate][ast]") { runFixture(fx("033_water_ast_ne_m05")); }
TEST_CASE("MUTATE-034_water_ast_ye_m00", "[ff][mutate][ast]") { runFixture(fx("034_water_ast_ye_m00")); }
TEST_CASE("MUTATE-035_water_ast_ne_m00", "[ff][mutate][ast]") { runFixture(fx("035_water_ast_ne_m00")); }
TEST_CASE("MUTATE-036_water_ast_v10", "[ff][mutate][ast]") { runFixture(fx("036_water_ast_v10")); }
TEST_CASE("MUTATE-037_water_ast_v05", "[ff][mutate][ast]") { runFixture(fx("037_water_ast_v05")); }
TEST_CASE("MUTATE-038_water_ast_v00", "[ff][mutate][ast]") { runFixture(fx("038_water_ast_v00")); }
TEST_CASE("MUTATE-039_water_ast_ye_mp05", "[ff][mutate][ast]") { runNoEmplarFixture(fx("039_water_ast_ye_mp05")); }
TEST_CASE("MUTATE-040_water_ast_ne_mp05", "[ff][mutate][ast]") { runFixture(fx("040_water_ast_ne_mp05")); }
TEST_CASE("MUTATE-041_water_adt_ye_p10", "[ff][mutate][adt]") { runFixture(fx("041_water_adt_ye_p10")); }
TEST_CASE("MUTATE-042_water_adt_ne_p10", "[ff][mutate][adt]") { runFixture(fx("042_water_adt_ne_p10")); }
TEST_CASE("MUTATE-043_water_adt_ye_p05", "[ff][mutate][adt]") { runFixture(fx("043_water_adt_ye_p05")); }
TEST_CASE("MUTATE-044_water_adt_ne_p05", "[ff][mutate][adt]") { runFixture(fx("044_water_adt_ne_p05")); }
TEST_CASE("MUTATE-045_water_adt_ye_p00", "[ff][mutate][adt]") { runFixture(fx("045_water_adt_ye_p00")); }
TEST_CASE("MUTATE-046_water_adt_ne_p00", "[ff][mutate][adt]") { runFixture(fx("046_water_adt_ne_p00")); }
TEST_CASE("MUTATE-047_water_adt_ye_mp05", "[ff][mutate][adt]") { runFixture(fx("047_water_adt_ye_mp05")); }
TEST_CASE("MUTATE-048_water_adt_ne_mp05", "[ff][mutate][adt]") { runFixture(fx("048_water_adt_ne_mp05")); }
TEST_CASE("MUTATE-049_water_qnt_ast_l10", "[ff][mutate][qnt]") { runFixture(fx("049_water_qnt_ast_l10")); }
TEST_CASE("MUTATE-050_water_qnt_ast_l05", "[ff][mutate][qnt]") { runFixture(fx("050_water_qnt_ast_l05")); }
TEST_CASE("MUTATE-051_water_qnt_ast_l00", "[ff][mutate][qnt]") { runFixture(fx("051_water_qnt_ast_l00")); }
TEST_CASE("MUTATE-052_water_qnt_adt_l10", "[ff][mutate][qnt]") { runFixture(fx("052_water_qnt_adt_l10")); }
TEST_CASE("MUTATE-053_water_qnt_adt_l05", "[ff][mutate][qnt]") { runFixture(fx("053_water_qnt_adt_l05")); }
TEST_CASE("MUTATE-054_water_qnt_adt_l00", "[ff][mutate][qnt]") { runFixture(fx("054_water_qnt_adt_l00")); }
TEST_CASE("MUTATE-055_water_exp_ast_l10", "[ff][mutate][exp]") { runFixture(fx("055_water_exp_ast_l10")); }
TEST_CASE("MUTATE-056_water_exp_ast_l05", "[ff][mutate][exp]") { runFixture(fx("056_water_exp_ast_l05")); }
TEST_CASE("MUTATE-057_water_exp_ast_l00", "[ff][mutate][exp]") { runFixture(fx("057_water_exp_ast_l00")); }
TEST_CASE("MUTATE-058_water_exp_adt_l10", "[ff][mutate][exp]") { runFixture(fx("058_water_exp_adt_l10")); }
TEST_CASE("MUTATE-059_water_exp_adt_l05", "[ff][mutate][exp]") { runFixture(fx("059_water_exp_adt_l05")); }
TEST_CASE("MUTATE-060_water_exp_adt_l00", "[ff][mutate][exp]") { runFixture(fx("060_water_exp_adt_l00")); }
TEST_CASE("MUTATE-061_water_inv_ast_l10", "[ff][mutate][inv]") { runFixture(fx("061_water_inv_ast_l10")); }
TEST_CASE("MUTATE-062_water_inv_ast_l05", "[ff][mutate][inv]") { runFixture(fx("062_water_inv_ast_l05")); }
TEST_CASE("MUTATE-063_water_inv_ast_l00", "[ff][mutate][inv]") { runFixture(fx("063_water_inv_ast_l00")); }
TEST_CASE("MUTATE-064_water_inv_adt_l10", "[ff][mutate][inv]") { runFixture(fx("064_water_inv_adt_l10")); }
TEST_CASE("MUTATE-065_water_inv_adt_l05", "[ff][mutate][inv]") { runFixture(fx("065_water_inv_adt_l05")); }
TEST_CASE("MUTATE-066_water_inv_adt_l00", "[ff][mutate][inv]") { runFixture(fx("066_water_inv_adt_l00")); }
TEST_CASE("MUTATE-067_water_exf_ast_m10", "[ff][mutate][exf]") { runFixture(fx("067_water_exf_ast_m10")); }
TEST_CASE("MUTATE-068_water_exf_ast_m05", "[ff][mutate][exf]") { runFixture(fx("068_water_exf_ast_m05")); }
TEST_CASE("MUTATE-069_water_exf_ast_m00", "[ff][mutate][exf]") { runFixture(fx("069_water_exf_ast_m00")); }
TEST_CASE("MUTATE-070_water_exf_adt_p10", "[ff][mutate][exf]") { runFixture(fx("070_water_exf_adt_p10")); }
TEST_CASE("MUTATE-071_water_exf_adt_p05", "[ff][mutate][exf]") { runFixture(fx("071_water_exf_adt_p05")); }
TEST_CASE("MUTATE-072_water_exf_adt_p00", "[ff][mutate][exf]") { runFixture(fx("072_water_exf_adt_p00")); }
TEST_CASE("MUTATE-073_water_exf_adt_mp05", "[ff][mutate][exf]") { runFixture(fx("073_water_exf_adt_mp05")); }
TEST_CASE("MUTATE-074_water_adt_ye_l10", "[ff][mutate][adt]") { runFixture(fx("074_water_adt_ye_l10")); }
TEST_CASE("MUTATE-075_water_adt_ne_l10", "[ff][mutate][adt]") { runFixture(fx("075_water_adt_ne_l10")); }
TEST_CASE("MUTATE-076_water_adt_ye_l05", "[ff][mutate][adt]") { runFixture(fx("076_water_adt_ye_l05")); }
TEST_CASE("MUTATE-077_water_adt_ne_l05", "[ff][mutate][adt]") { runFixture(fx("077_water_adt_ne_l05")); }
TEST_CASE("MUTATE-078_water_adt_ye_l00", "[ff][mutate][adt]") { runFixture(fx("078_water_adt_ye_l00")); }
TEST_CASE("MUTATE-079_water_adt_ne_l00", "[ff][mutate][adt]") { runFixture(fx("079_water_adt_ne_l00")); }
TEST_CASE("MUTATE-080_water_qnt_adt_l10", "[ff][mutate][legskip]") { runLegSkipFixture(fx("080_water_qnt_adt_l10"), false, true); }
TEST_CASE("MUTATE-081_water_qnt_adt_l00", "[ff][mutate][legskip]") { runLegSkipFixture(fx("081_water_qnt_adt_l00"), true, false); }
TEST_CASE("MUTATE-082_water_rels_ye_l100", "[ff][mutate][rels]") { runFixture(fx("082_water_rels_ye_l100")); }
TEST_CASE("MUTATE-083_water_rels_ye_l085", "[ff][mutate][rels]") { runFixture(fx("083_water_rels_ye_l085")); }
TEST_CASE("MUTATE-084_water_rels_ye_l070", "[ff][mutate][rels]") { runFixture(fx("084_water_rels_ye_l070")); }
TEST_CASE("MUTATE-085_water_rels_ye_l050", "[ff][mutate][rels]") { runFixture(fx("085_water_rels_ye_l050")); }
TEST_CASE("MUTATE-086_water_rels_ye_l030", "[ff][mutate][rels]") { runFixture(fx("086_water_rels_ye_l030")); }
TEST_CASE("MUTATE-087_water_rels_ye_l015", "[ff][mutate][rels]") { runFixture(fx("087_water_rels_ye_l015")); }
TEST_CASE("MUTATE-088_water_rels_ye_l000", "[ff][mutate][rels]") { runFixture(fx("088_water_rels_ye_l000")); }
TEST_CASE("MUTATE-098_water_lmda_p_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("098_water_lmda_p_l06")); }
TEST_CASE("MUTATE-100_water_lmda_ep_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("100_water_lmda_ep_l06")); }
TEST_CASE("MUTATE-102_water_lmda_pv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("102_water_lmda_pv_l06")); }
TEST_CASE("MUTATE-103_water_lmda_epv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("103_water_lmda_epv_l06")); }
TEST_CASE("MUTATE-104_water_dlmda_e_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("104_water_dlmda_e_l06")); }
TEST_CASE("MUTATE-105_water_dlmda_p_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("105_water_dlmda_p_l06")); }
TEST_CASE("MUTATE-106_water_dlmda_v_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("106_water_dlmda_v_l06")); }
TEST_CASE("MUTATE-107_water_dlmda_ep_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("107_water_dlmda_ep_l06")); }
TEST_CASE("MUTATE-108_water_dlmda_ev_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("108_water_dlmda_ev_l06")); }
TEST_CASE("MUTATE-109_water_dlmda_pv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("109_water_dlmda_pv_l06")); }
TEST_CASE("MUTATE-110_water_dlmda_epv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("110_water_dlmda_epv_l06")); }
TEST_CASE("MUTATE-111_water_rels_ye_vdwm_l030", "[ff][mutate][rels]") { runFixture(fx("111_water_rels_ye_vdwm_l030")); }
TEST_CASE("MUTATE-112_water_rels_ye_lig1_l070", "[ff][mutate][rels]") { runFixture(fx("112_water_rels_ye_lig1_l070")); }
TEST_CASE("MUTATE-113_water_lmda_ast_epin_l05", "[ff][mutate][pin]") { runFixture(fx("113_water_lmda_ast_epin_l05")); }
TEST_CASE("MUTATE-114_water_lmda_ast_vpin_l05", "[ff][mutate][pin]") { runFixture(fx("114_water_lmda_ast_vpin_l05")); }
TEST_CASE("MUTATE-115_water_lmda_adt_vpin_l06", "[ff][mutate][pin]") { runFixture(fx("115_water_lmda_adt_vpin_l06")); }
TEST_CASE("MUTATE-116_water_rels_ye_vdwm_exp_l050", "[ff][mutate][rels]") { runFixture(fx("116_water_rels_ye_vdwm_exp_l050")); }
TEST_CASE("MUTATE-117_water_rels_ye_lig2_exp_l030", "[ff][mutate][rels]") { runFixture(fx("117_water_rels_ye_lig2_exp_l030")); }
TEST_CASE("MUTATE-118_water_rels_ye_lig1_inv_l070", "[ff][mutate][rels]") { runFixture(fx("118_water_rels_ye_lig1_inv_l070")); }
TEST_CASE("MUTATE-119_water_rels_ye_vdwm_vx3_l050", "[ff][mutate][rels]") { runFixture(fx("119_water_rels_ye_vdwm_vx3_l050")); }
TEST_CASE("MUTATE-120_water_rels_ye_lig1_ex3_l085", "[ff][mutate][rels]") { runFixture(fx("120_water_rels_ye_lig1_ex3_l085")); }
TEST_CASE("MUTATE-121_water_rels_ye_lig2_ix2_l015", "[ff][mutate][rels]") { runFixture(fx("121_water_rels_ye_lig2_ix2_l015")); }
TEST_CASE("MUTATE-122_water_ast_v05_annihilate", "[ff][mutate][ast]") { runFixture(fx("122_water_ast_v05_annihilate")); }
TEST_CASE("MUTATE-123_water_exf_adt_l10", "[ff][mutate][exf]") { runFixture(fx("123_water_exf_adt_l10")); }
TEST_CASE("MUTATE-124_water_exf_adt_l05", "[ff][mutate][exf]") { runFixture(fx("124_water_exf_adt_l05")); }
TEST_CASE("MUTATE-125_water_exf_adt_l00", "[ff][mutate][exf]") { runFixture(fx("125_water_exf_adt_l00")); }
TEST_CASE("MUTATE-126_water_ast_ye_l10", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("126_water_ast_ye_l10")); }
TEST_CASE("MUTATE-127_water_ast_ne_l10", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("127_water_ast_ne_l10")); }
TEST_CASE("MUTATE-128_water_ast_ye_l05", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("128_water_ast_ye_l05")); }
TEST_CASE("MUTATE-129_water_ast_ne_l05", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("129_water_ast_ne_l05")); }
TEST_CASE("MUTATE-130_water_ast_ye_l00", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("130_water_ast_ye_l00")); }
TEST_CASE("MUTATE-131_water_ast_ne_l00", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("131_water_ast_ne_l00")); }
TEST_CASE("MUTATE-132_water_exf_ast_l10", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("132_water_exf_ast_l10")); }
TEST_CASE("MUTATE-133_water_exf_ast_l05", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("133_water_exf_ast_l05")); }
TEST_CASE("MUTATE-134_water_exf_ast_l00", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("134_water_exf_ast_l00")); }
TEST_CASE("MUTATE-135_water_apm_ast_vpin_l05", "[ff][mutate][apm][pin]") { runFixture(fx("135_water_apm_ast_vpin_l05")); }
TEST_CASE("MUTATE-136_water_apm_ast_epin_l00", "[ff][mutate][apm][pin]") { runFixture(fx("136_water_apm_ast_epin_l00")); }
TEST_CASE("MUTATE-137_water_apm_ast_epin_l05", "[ff][mutate][apm][pin]") { runFixture(fx("137_water_apm_ast_epin_l05")); }
TEST_CASE("MUTATE-138_water_apm_ast_epin_l10", "[ff][mutate][apm][pin]") { runFixture(fx("138_water_apm_ast_epin_l10")); }
TEST_CASE("MUTATE-139_water_vsoft_l10", "[ff][mutate][vsoft]") { runFixture(fx("139_water_vsoft_l10")); }
TEST_CASE("MUTATE-140_water_vsoft_l05", "[ff][mutate][vsoft]") { runFixture(fx("140_water_vsoft_l05")); }
TEST_CASE("MUTATE-141_water_vsoft_l00", "[ff][mutate][vsoft]") { runFixture(fx("141_water_vsoft_l00")); }
TEST_CASE("MUTATE-142_water_rels_st_l085", "[ff][mutate][rels][astpol][emplar]") { runEmplarAstFixture(fx("142_water_rels_st_l085")); }
TEST_CASE("MUTATE-143_water_rels_st_lig2_exp_l030", "[ff][mutate][rels][astpol][emplar]") { runEmplarAstFixture(fx("143_water_rels_st_lig2_exp_l030")); }
TEST_CASE("MUTATE-144_water_rels_nolmda", "[ff][mutate][rels]") { runFixture(fx("144_water_rels_nolmda")); }
TEST_CASE("MUTATE-145_trpcage_chiral_m05", "[ff][mutate][chiral]") { runChiralFixture(fx("145_trpcage_chiral_m05")); }
TEST_CASE("MUTATE-146_g3_ast_ye_l10", "[ff][mutate][g3]") { runFixture(fx("146_g3_ast_ye_l10")); }
TEST_CASE("MUTATE-147_g3_ast_ye_l05", "[ff][mutate][g3]") { runFixture(fx("147_g3_ast_ye_l05")); }
TEST_CASE("MUTATE-148_g3_ast_annih_l05", "[ff][mutate][g3]") { runFixture(fx("148_g3_ast_annih_l05")); }
TEST_CASE("MUTATE-149_g3_ast_nobox_l05", "[ff][mutate][g3]") { runFixture(fx("149_g3_ast_nobox_l05")); }
TEST_CASE("MUTATE-150_water_ast_vcorr_annih_l05", "[ff][mutate][hal]") { runFixture(fx("150_water_ast_vcorr_annih_l05")); }
TEST_CASE("MUTATE-151_water_ast_mono_l05", "[ff][mutate][hal]") { runFixture(fx("151_water_ast_mono_l05")); }
TEST_CASE("MUTATE-152_water_ast_tric_l05", "[ff][mutate][hal]") { runFixture(fx("152_water_ast_tric_l05")); }
TEST_CASE("MUTATE-153_water_rels_ye_vdwm_lig2t_l040", "[ff][mutate][hal]") { runFixture(fx("153_water_rels_ye_vdwm_lig2t_l040")); }
TEST_CASE("MUTATE-154_water_ast_ne_mcut_l05", "[ff][mutate][mpole4]") { runFixture(fx("154_water_ast_ne_mcut_l05")); }
TEST_CASE("MUTATE-155_g3_rels_lig1_l085", "[ff][mutate][mpole4]") { runFixture(fx("155_g3_rels_lig1_l085")); }
TEST_CASE("MUTATE-156_g3_rels_lig2_l015", "[ff][mutate][mpole4]") { runFixture(fx("156_g3_rels_lig2_l015")); }
TEST_CASE("MUTATE-157_water_rels_ne_lig2_l015", "[ff][mutate][mpole4]") { runFixture(fx("157_water_rels_ne_lig2_l015")); }
TEST_CASE("MUTATE-158_frames_ast_ye_l05", "[ff][mutate][frames]") { runFixture(fx("158_frames_ast_ye_l05")); }
TEST_CASE("MUTATE-159_frames_ast_nobox_l05", "[ff][mutate][frames]") { runFixture(fx("159_frames_ast_nobox_l05")); }
TEST_CASE("MUTATE-160_chig_ast_nobox_l05", "[ff][mutate][mirror]") { runFixture(fx("160_chig_ast_nobox_l05")); }
TEST_CASE("MUTATE-161_chigm_ast_nobox_l05", "[ff][mutate][mirror]") { runFixture(fx("161_chigm_ast_nobox_l05")); }
TEST_CASE("MUTATE-162_chig_ast_ye_l05", "[ff][mutate][mirror]") { runFixture(fx("162_chig_ast_ye_l05")); }
TEST_CASE("MUTATE-163_chigm_ast_ye_l05", "[ff][mutate][mirror]") { runFixture(fx("163_chigm_ast_ye_l05")); }
TEST_CASE("MUTATE-164_water_ast_ne_mcut_d1_l05", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("164_water_ast_ne_mcut_d1_l05")); }
TEST_CASE("MUTATE-165_g3_ast_d1_l05", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("165_g3_ast_d1_l05")); }
TEST_CASE("MUTATE-166_g3_ast_d1_l00", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("166_g3_ast_d1_l00")); }
TEST_CASE("MUTATE-167_g3_rels_lig1_d1_l085", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("167_g3_rels_lig1_d1_l085")); }
TEST_CASE("MUTATE-168_g3_rels_lig2_d1_l015", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("168_g3_rels_lig2_d1_l015")); }
TEST_CASE("MUTATE-169_water_adt_d1_x2_l06", "[ff][mutate][deriv1]") { runFixture(fx("169_water_adt_d1_x2_l06")); }
TEST_CASE("MUTATE-170_water_ast_vcorr_annih_d1_l05", "[ff][mutate][deriv1]") { runFixture(fx("170_water_ast_vcorr_annih_d1_l05")); }
TEST_CASE("MUTATE-171_water_rels_ye_vdwm_d1_l040", "[ff][mutate][deriv1]") { runFixture(fx("171_water_rels_ye_vdwm_d1_l040")); }
TEST_CASE("MUTATE-172_water_vsoft_n1_d1_l00", "[ff][mutate][deriv1]") { runFixture(fx("172_water_vsoft_n1_d1_l00")); }
TEST_CASE("MUTATE-173_water_vsoft_n1_d1_l005", "[ff][mutate][deriv1]") { runFixture(fx("173_water_vsoft_n1_d1_l005")); }
TEST_CASE("MUTATE-174_water_rels_vdwm_n1_d1_l10", "[ff][mutate][deriv1]") { runFixture(fx("174_water_rels_vdwm_n1_d1_l10")); }
TEST_CASE("MUTATE-175_water_vsoft_n15_ti_l00", "[ff][mutate][scexp]") { runScexpTiFixture(fx("175_water_vsoft_n15_ti_l00")); }
TEST_CASE("MUTATE-176_water_rels_st_ne_l085", "[ff][mutate][noewald][astpol][emplar]") { runEmplarAstFixture(fx("176_water_rels_st_ne_l085")); }
TEST_CASE("MUTATE-177_water_rels_ne_l085", "[ff][mutate][noewald]") { runFixture(fx("177_water_rels_ne_l085")); }
TEST_CASE("MUTATE-178_water_adt_d1_ne_l06", "[ff][mutate][noewald]") { runFixture(fx("178_water_adt_d1_ne_l06")); }
TEST_CASE("MUTATE-179_ionwat_ast_l05", "[ff][mutate][ion][astpol][emplar]") { runEmplarAstFixture(fx("179_ionwat_ast_l05")); }

// The fused kernel on its own. The ordinary case of each fixture asks for
// interaction counts and so goes through the separate empole and epolar
// kernels; these drop the counts, where emplar fuses the two. Both are checked
// against the same reference, so the fused kernel has to agree with the split
// one and with Tinker. The fixtures have no lambda derivative and one lambda
// value for electrostatics and polarization, so plain emplar takes over.
TEST_CASE("MUTATE-EMPLAR-001_water_ye_m10", "[ff][mutate][emplar]") { runFixture(fx("001_water_ye_m10"), Fuse::Require); }
TEST_CASE("MUTATE-EMPLAR-003_water_ye_m05", "[ff][mutate][emplar]") { runFixture(fx("003_water_ye_m05"), Fuse::Require); }
TEST_CASE("MUTATE-EMPLAR-090_water_lmda_ast_e05", "[ff][mutate][emplar]") { runFixture(fx("090_water_lmda_ast_e05"), Fuse::Require); }

TEST_CASE("MUTATE-TI-050_water_qnt_ast_l05", "[ff][mutate][ti][ast]") { runFixture(fx("050_water_qnt_ast_l05"), Fuse::Off, LmdaMode::ThermIntg); }
TEST_CASE("MUTATE-TI-053_water_qnt_adt_l05", "[ff][mutate][ti][adt]") { runFixture(fx("053_water_qnt_adt_l05"), Fuse::Off, LmdaMode::ThermIntg); }
TEST_CASE("MUTATE-TI-116_water_rels_ye_vdwm_exp_l050", "[ff][mutate][ti][rels]") { runFixture(fx("116_water_rels_ye_vdwm_exp_l050"), Fuse::Off, LmdaMode::ThermIntg); }

// Every lambda derivative fixture once more without analysis, as dynamics runs
// it: the terms add into the shared lambda derivative buffers, and only the
// totals are there to compare. The fixtures run fused above already run this way.
TEST_CASE("MUTATE-DYN-030_water_ast_ye_m10", "[ff][mutate][dyn]") { runFixture(fx("030_water_ast_ye_m10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-031_water_ast_ne_m10", "[ff][mutate][dyn]") { runFixture(fx("031_water_ast_ne_m10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-032_water_ast_ye_m05", "[ff][mutate][dyn]") { runFixture(fx("032_water_ast_ye_m05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-033_water_ast_ne_m05", "[ff][mutate][dyn]") { runFixture(fx("033_water_ast_ne_m05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-034_water_ast_ye_m00", "[ff][mutate][dyn]") { runFixture(fx("034_water_ast_ye_m00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-035_water_ast_ne_m00", "[ff][mutate][dyn]") { runFixture(fx("035_water_ast_ne_m00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-036_water_ast_v10", "[ff][mutate][dyn]") { runFixture(fx("036_water_ast_v10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-037_water_ast_v05", "[ff][mutate][dyn]") { runFixture(fx("037_water_ast_v05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-038_water_ast_v00", "[ff][mutate][dyn]") { runFixture(fx("038_water_ast_v00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-040_water_ast_ne_mp05", "[ff][mutate][dyn]") { runFixture(fx("040_water_ast_ne_mp05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-041_water_adt_ye_p10", "[ff][mutate][dyn]") { runFixture(fx("041_water_adt_ye_p10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-042_water_adt_ne_p10", "[ff][mutate][dyn]") { runFixture(fx("042_water_adt_ne_p10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-043_water_adt_ye_p05", "[ff][mutate][dyn]") { runFixture(fx("043_water_adt_ye_p05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-044_water_adt_ne_p05", "[ff][mutate][dyn]") { runFixture(fx("044_water_adt_ne_p05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-045_water_adt_ye_p00", "[ff][mutate][dyn]") { runFixture(fx("045_water_adt_ye_p00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-046_water_adt_ne_p00", "[ff][mutate][dyn]") { runFixture(fx("046_water_adt_ne_p00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-047_water_adt_ye_mp05", "[ff][mutate][dyn]") { runFixture(fx("047_water_adt_ye_mp05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-048_water_adt_ne_mp05", "[ff][mutate][dyn]") { runFixture(fx("048_water_adt_ne_mp05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-049_water_qnt_ast_l10", "[ff][mutate][dyn]") { runFixture(fx("049_water_qnt_ast_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-050_water_qnt_ast_l05", "[ff][mutate][dyn]") { runFixture(fx("050_water_qnt_ast_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-051_water_qnt_ast_l00", "[ff][mutate][dyn]") { runFixture(fx("051_water_qnt_ast_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-052_water_qnt_adt_l10", "[ff][mutate][dyn]") { runFixture(fx("052_water_qnt_adt_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-053_water_qnt_adt_l05", "[ff][mutate][dyn]") { runFixture(fx("053_water_qnt_adt_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-054_water_qnt_adt_l00", "[ff][mutate][dyn]") { runFixture(fx("054_water_qnt_adt_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-055_water_exp_ast_l10", "[ff][mutate][dyn]") { runFixture(fx("055_water_exp_ast_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-056_water_exp_ast_l05", "[ff][mutate][dyn]") { runFixture(fx("056_water_exp_ast_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-057_water_exp_ast_l00", "[ff][mutate][dyn]") { runFixture(fx("057_water_exp_ast_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-058_water_exp_adt_l10", "[ff][mutate][dyn]") { runFixture(fx("058_water_exp_adt_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-059_water_exp_adt_l05", "[ff][mutate][dyn]") { runFixture(fx("059_water_exp_adt_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-060_water_exp_adt_l00", "[ff][mutate][dyn]") { runFixture(fx("060_water_exp_adt_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-061_water_inv_ast_l10", "[ff][mutate][dyn]") { runFixture(fx("061_water_inv_ast_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-062_water_inv_ast_l05", "[ff][mutate][dyn]") { runFixture(fx("062_water_inv_ast_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-063_water_inv_ast_l00", "[ff][mutate][dyn]") { runFixture(fx("063_water_inv_ast_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-064_water_inv_adt_l10", "[ff][mutate][dyn]") { runFixture(fx("064_water_inv_adt_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-065_water_inv_adt_l05", "[ff][mutate][dyn]") { runFixture(fx("065_water_inv_adt_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-066_water_inv_adt_l00", "[ff][mutate][dyn]") { runFixture(fx("066_water_inv_adt_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-067_water_exf_ast_m10", "[ff][mutate][dyn]") { runFixture(fx("067_water_exf_ast_m10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-068_water_exf_ast_m05", "[ff][mutate][dyn]") { runFixture(fx("068_water_exf_ast_m05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-069_water_exf_ast_m00", "[ff][mutate][dyn]") { runFixture(fx("069_water_exf_ast_m00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-070_water_exf_adt_p10", "[ff][mutate][dyn]") { runFixture(fx("070_water_exf_adt_p10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-071_water_exf_adt_p05", "[ff][mutate][dyn]") { runFixture(fx("071_water_exf_adt_p05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-072_water_exf_adt_p00", "[ff][mutate][dyn]") { runFixture(fx("072_water_exf_adt_p00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-073_water_exf_adt_mp05", "[ff][mutate][dyn]") { runFixture(fx("073_water_exf_adt_mp05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-074_water_adt_ye_l10", "[ff][mutate][dyn]") { runFixture(fx("074_water_adt_ye_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-075_water_adt_ne_l10", "[ff][mutate][dyn]") { runFixture(fx("075_water_adt_ne_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-076_water_adt_ye_l05", "[ff][mutate][dyn]") { runFixture(fx("076_water_adt_ye_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-077_water_adt_ne_l05", "[ff][mutate][dyn]") { runFixture(fx("077_water_adt_ne_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-078_water_adt_ye_l00", "[ff][mutate][dyn]") { runFixture(fx("078_water_adt_ye_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-079_water_adt_ne_l00", "[ff][mutate][dyn]") { runFixture(fx("079_water_adt_ne_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-080_water_qnt_adt_l10", "[ff][mutate][dyn]") { runFixture(fx("080_water_qnt_adt_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-081_water_qnt_adt_l00", "[ff][mutate][dyn]") { runFixture(fx("081_water_qnt_adt_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-082_water_rels_ye_l100", "[ff][mutate][dyn]") { runFixture(fx("082_water_rels_ye_l100"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-083_water_rels_ye_l085", "[ff][mutate][dyn]") { runFixture(fx("083_water_rels_ye_l085"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-084_water_rels_ye_l070", "[ff][mutate][dyn]") { runFixture(fx("084_water_rels_ye_l070"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-085_water_rels_ye_l050", "[ff][mutate][dyn]") { runFixture(fx("085_water_rels_ye_l050"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-086_water_rels_ye_l030", "[ff][mutate][dyn]") { runFixture(fx("086_water_rels_ye_l030"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-087_water_rels_ye_l015", "[ff][mutate][dyn]") { runFixture(fx("087_water_rels_ye_l015"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-088_water_rels_ye_l000", "[ff][mutate][dyn]") { runFixture(fx("088_water_rels_ye_l000"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-104_water_dlmda_e_l06", "[ff][mutate][dyn]") { runFixture(fx("104_water_dlmda_e_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-105_water_dlmda_p_l06", "[ff][mutate][dyn]") { runFixture(fx("105_water_dlmda_p_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-106_water_dlmda_v_l06", "[ff][mutate][dyn]") { runFixture(fx("106_water_dlmda_v_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-107_water_dlmda_ep_l06", "[ff][mutate][dyn]") { runFixture(fx("107_water_dlmda_ep_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-108_water_dlmda_ev_l06", "[ff][mutate][dyn]") { runFixture(fx("108_water_dlmda_ev_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-109_water_dlmda_pv_l06", "[ff][mutate][dyn]") { runFixture(fx("109_water_dlmda_pv_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-110_water_dlmda_epv_l06", "[ff][mutate][dyn]") { runFixture(fx("110_water_dlmda_epv_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-111_water_rels_ye_vdwm_l030", "[ff][mutate][dyn]") { runFixture(fx("111_water_rels_ye_vdwm_l030"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-112_water_rels_ye_lig1_l070", "[ff][mutate][dyn]") { runFixture(fx("112_water_rels_ye_lig1_l070"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-113_water_lmda_ast_epin_l05", "[ff][mutate][dyn]") { runFixture(fx("113_water_lmda_ast_epin_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-114_water_lmda_ast_vpin_l05", "[ff][mutate][dyn]") { runFixture(fx("114_water_lmda_ast_vpin_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-115_water_lmda_adt_vpin_l06", "[ff][mutate][dyn]") { runFixture(fx("115_water_lmda_adt_vpin_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-116_water_rels_ye_vdwm_exp_l050", "[ff][mutate][dyn]") { runFixture(fx("116_water_rels_ye_vdwm_exp_l050"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-117_water_rels_ye_lig2_exp_l030", "[ff][mutate][dyn]") { runFixture(fx("117_water_rels_ye_lig2_exp_l030"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-118_water_rels_ye_lig1_inv_l070", "[ff][mutate][dyn]") { runFixture(fx("118_water_rels_ye_lig1_inv_l070"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-119_water_rels_ye_vdwm_vx3_l050", "[ff][mutate][dyn]") { runFixture(fx("119_water_rels_ye_vdwm_vx3_l050"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-120_water_rels_ye_lig1_ex3_l085", "[ff][mutate][dyn]") { runFixture(fx("120_water_rels_ye_lig1_ex3_l085"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-121_water_rels_ye_lig2_ix2_l015", "[ff][mutate][dyn]") { runFixture(fx("121_water_rels_ye_lig2_ix2_l015"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-122_water_ast_v05_annihilate", "[ff][mutate][dyn]") { runFixture(fx("122_water_ast_v05_annihilate"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-123_water_exf_adt_l10", "[ff][mutate][dyn]") { runFixture(fx("123_water_exf_adt_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-124_water_exf_adt_l05", "[ff][mutate][dyn]") { runFixture(fx("124_water_exf_adt_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-125_water_exf_adt_l00", "[ff][mutate][dyn]") { runFixture(fx("125_water_exf_adt_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-135_water_apm_ast_vpin_l05", "[ff][mutate][dyn]") { runFixture(fx("135_water_apm_ast_vpin_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-136_water_apm_ast_epin_l00", "[ff][mutate][dyn]") { runFixture(fx("136_water_apm_ast_epin_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-137_water_apm_ast_epin_l05", "[ff][mutate][dyn]") { runFixture(fx("137_water_apm_ast_epin_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-138_water_apm_ast_epin_l10", "[ff][mutate][dyn]") { runFixture(fx("138_water_apm_ast_epin_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-139_water_vsoft_l10", "[ff][mutate][dyn]") { runFixture(fx("139_water_vsoft_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-140_water_vsoft_l05", "[ff][mutate][dyn]") { runFixture(fx("140_water_vsoft_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-141_water_vsoft_l00", "[ff][mutate][dyn]") { runFixture(fx("141_water_vsoft_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-144_water_rels_nolmda", "[ff][mutate][dyn]") { runFixture(fx("144_water_rels_nolmda"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-146_g3_ast_ye_l10", "[ff][mutate][dyn]") { runFixture(fx("146_g3_ast_ye_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-147_g3_ast_ye_l05", "[ff][mutate][dyn]") { runFixture(fx("147_g3_ast_ye_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-148_g3_ast_annih_l05", "[ff][mutate][dyn]") { runFixture(fx("148_g3_ast_annih_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-149_g3_ast_nobox_l05", "[ff][mutate][dyn]") { runFixture(fx("149_g3_ast_nobox_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-150_water_ast_vcorr_annih_l05", "[ff][mutate][dyn]") { runFixture(fx("150_water_ast_vcorr_annih_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-151_water_ast_mono_l05", "[ff][mutate][dyn]") { runFixture(fx("151_water_ast_mono_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-152_water_ast_tric_l05", "[ff][mutate][dyn]") { runFixture(fx("152_water_ast_tric_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-153_water_rels_ye_vdwm_lig2t_l040", "[ff][mutate][dyn]") { runFixture(fx("153_water_rels_ye_vdwm_lig2t_l040"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-154_water_ast_ne_mcut_l05", "[ff][mutate][dyn]") { runFixture(fx("154_water_ast_ne_mcut_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-155_g3_rels_lig1_l085", "[ff][mutate][dyn]") { runFixture(fx("155_g3_rels_lig1_l085"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-156_g3_rels_lig2_l015", "[ff][mutate][dyn]") { runFixture(fx("156_g3_rels_lig2_l015"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-157_water_rels_ne_lig2_l015", "[ff][mutate][dyn]") { runFixture(fx("157_water_rels_ne_lig2_l015"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-158_frames_ast_ye_l05", "[ff][mutate][dyn]") { runFixture(fx("158_frames_ast_ye_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-159_frames_ast_nobox_l05", "[ff][mutate][dyn]") { runFixture(fx("159_frames_ast_nobox_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-160_chig_ast_nobox_l05", "[ff][mutate][dyn]") { runFixture(fx("160_chig_ast_nobox_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-161_chigm_ast_nobox_l05", "[ff][mutate][dyn]") { runFixture(fx("161_chigm_ast_nobox_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-162_chig_ast_ye_l05", "[ff][mutate][dyn]") { runFixture(fx("162_chig_ast_ye_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-163_chigm_ast_ye_l05", "[ff][mutate][dyn]") { runFixture(fx("163_chigm_ast_ye_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-169_water_adt_d1_x2_l06", "[ff][mutate][dyn]") { runFixture(fx("169_water_adt_d1_x2_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-170_water_ast_vcorr_annih_d1_l05", "[ff][mutate][dyn]") { runFixture(fx("170_water_ast_vcorr_annih_d1_l05"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-171_water_rels_ye_vdwm_d1_l040", "[ff][mutate][dyn]") { runFixture(fx("171_water_rels_ye_vdwm_d1_l040"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-172_water_vsoft_n1_d1_l00", "[ff][mutate][dyn]") { runFixture(fx("172_water_vsoft_n1_d1_l00"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-173_water_vsoft_n1_d1_l005", "[ff][mutate][dyn]") { runFixture(fx("173_water_vsoft_n1_d1_l005"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-174_water_rels_vdwm_n1_d1_l10", "[ff][mutate][dyn]") { runFixture(fx("174_water_rels_vdwm_n1_d1_l10"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-177_water_rels_ne_l085", "[ff][mutate][dyn]") { runFixture(fx("177_water_rels_ne_l085"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-178_water_adt_d1_ne_l06", "[ff][mutate][dyn]") { runFixture(fx("178_water_adt_d1_ne_l06"), Fuse::Auto); }
TEST_CASE("MUTATE-DYN-TI-050_water_qnt_ast_l05", "[ff][mutate][dyn][ti]") { runFixture(fx("050_water_qnt_ast_l05"), Fuse::Auto, LmdaMode::ThermIntg); }
TEST_CASE("MUTATE-DYN-TI-053_water_qnt_adt_l05", "[ff][mutate][dyn][ti]") { runFixture(fx("053_water_qnt_adt_l05"), Fuse::Auto, LmdaMode::ThermIntg); }
TEST_CASE("MUTATE-DYN-TI-116_water_rels_ye_vdwm_exp_l050", "[ff][mutate][dyn][ti]") { runFixture(fx("116_water_rels_ye_vdwm_exp_l050"), Fuse::Auto, LmdaMode::ThermIntg); }

TEST_CASE("MUTATE-gate", "[ff][mutate][gate]") { runGateFixture(fx("083_water_rels_ye_l085")); }

TEST_CASE("MUTATE-flags", "[ff][mutate][flags]")
{
   runFlagsFixture(fx("095_water_lmda_mp05"), "\nLAMBDA-DERIV\n", false, false, false);
   runFlagsFixture(fx("095_water_lmda_mp05"), "\nLAMBDA-DERIV2\n", true, true, false);
   runFlagsFixture(fx("142_water_rels_st_l085"), "", false, false, true);
   runFlagsFixture(fx("083_water_rels_ye_l085"), "", true, true, true);
}

TEST_CASE("MUTATE-flat-single", "[ff][mutate][rels][flat]") { runFlatFixture(fx("142_water_rels_st_l085"), false); }
TEST_CASE("MUTATE-flat-single-fused", "[ff][mutate][rels][flat][emplar]") { runFlatFixture(fx("142_water_rels_st_l085"), true); }
TEST_CASE("MUTATE-flat-dual", "[ff][mutate][rels][flat]") { runFlatFixture(fx("083_water_rels_ye_l085"), false); }
TEST_CASE("MUTATE-flat-single-ne", "[ff][mutate][rels][flat][noewald]") { runFlatFixture(fx("176_water_rels_st_ne_l085"), false); }
TEST_CASE("MUTATE-flat-single-ne-fused", "[ff][mutate][rels][flat][noewald][emplar]") { runFlatFixture(fx("176_water_rels_st_ne_l085"), true); }
TEST_CASE("MUTATE-flat-dual-ne", "[ff][mutate][rels][flat][noewald]") { runFlatFixture(fx("177_water_rels_ne_l085"), false); }
#endif

#endif
