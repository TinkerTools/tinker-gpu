#include "ff/amoeba/emplar.h"
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

#include <tinker/detail/atoms.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace tinker;

#if TINKER_GPULANG_CUDA

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
   const char* cat;
   const char* ref = nullptr; ///< Reference borrowed from another fixture, if any.
   Tols tol = {};             ///< Tolerances that differ from the defaults.
};

const Fixture kFixtures[] = {
   {"001_water_ye_m10", "water", true, true, false, false, "mv"},
   {"002_water_ne_m10", "water", true, true, false, false, "mv"},
   {"003_water_ye_m05", "water", true, true, false, false, "mv"},
   {"004_water_ne_m05", "water", true, true, false, false, "mv"},
   {"005_water_ye_m00", "water", true, true, false, false, "mv"},
   {"006_water_ne_m00", "water", true, true, false, false, "mv"},
   {"007_water_v10", "water", false, false, true, false, "mv"},
   {"008_water_v05", "water", false, false, true, false, "mv"},
   {"009_water_v00", "water", false, false, true, false, "mv"},
   {"010_water_ye_m10", "water", true, false, false, false, "mp"},
   {"011_water_ne_m10", "water", true, false, false, false, "mp"},
   {"012_water_ye_m05", "water", true, false, false, false, "mp"},
   {"013_water_ne_m05", "water", true, false, false, false, "mp"},
   {"014_water_ye_m00", "water", true, false, false, false, "mp"},
   {"015_water_ne_m00", "water", true, false, false, false, "mp"},
   {"016_water_ye_p10", "water", false, true, false, false, "mp"},
   {"017_water_ne_p10", "water", false, true, false, false, "mp"},
   {"018_water_ye_p05", "water", false, true, false, false, "mp"},
   {"019_water_ne_p05", "water", false, true, false, false, "mp"},
   {"020_water_ye_p00", "water", false, true, false, false, "mp"},
   {"021_water_ne_p00", "water", false, true, false, false, "mp"},
   {"022_water_ye_m10p05", "water", true, true, false, false, "mp"},
   {"023_water_ne_m10p05", "water", true, true, false, false, "mp"},
   {"024_water_ye_m05p10", "water", true, true, false, false, "mp"},
   {"025_water_ne_m05p10", "water", true, true, false, false, "mp"},
   {"026_water_ye_m05p00", "water", true, true, false, false, "mp"},
   {"027_water_ne_m05p00", "water", true, true, false, false, "mp"},
   {"028_water_ye_m00p05", "water", true, true, false, false, "mp"},
   {"029_water_ne_m00p05", "water", true, true, false, false, "mp"},
   {"030_water_ast_ye_m10", "water2", true, false, false, true, "ast"},
   {"031_water_ast_ne_m10", "water2", true, false, false, true, "ast"},
   {"032_water_ast_ye_m05", "water2", true, false, false, true, "ast"},
   {"033_water_ast_ne_m05", "water2", true, false, false, true, "ast"},
   {"034_water_ast_ye_m00", "water2", true, false, false, true, "ast"},
   {"035_water_ast_ne_m00", "water2", true, false, false, true, "ast"},
   {"036_water_ast_v10", "water2", false, false, true, true, "ast"},
   {"037_water_ast_v05", "water2", false, false, true, true, "ast"},
   {"038_water_ast_v00", "water2", false, false, true, true, "ast"},
   {"039_water_ast_ye_mp05", "water2", true, true, false, true, "ast"},
   {"040_water_ast_ne_mp05", "water2", true, true, false, true, "ast"},
   {"047_water_adt_ye_p10", "water2", false, true, false, true, "adt"},
   {"048_water_adt_ne_p10", "water2", false, true, false, true, "adt"},
   {"049_water_adt_ye_p05", "water2", false, true, false, true, "adt"},
   {"050_water_adt_ne_p05", "water2", false, true, false, true, "adt"},
   {"051_water_adt_ye_p00", "water2", false, true, false, true, "adt"},
   {"052_water_adt_ne_p00", "water2", false, true, false, true, "adt"},
   {"056_water_adt_ye_mp05", "water2", true, true, false, true, "adt"},
   {"057_water_adt_ne_mp05", "water2", true, true, false, true, "adt"},
   {"075_water_qnt_ast_l10", "water2", true, true, true, true, "qnt"},
   {"076_water_qnt_ast_l05", "water2", true, true, true, true, "qnt"},
   {"077_water_qnt_ast_l00", "water2", true, true, true, true, "qnt"},
   {"078_water_qnt_adt_l10", "water2", true, true, false, true, "qnt"},
   {"079_water_qnt_adt_l05", "water2", true, true, false, true, "qnt"},
   {"080_water_qnt_adt_l00", "water2", true, true, false, true, "qnt"},
   {"084_water_exp_ast_l10", "water2", true, true, true, true, "exp"},
   {"085_water_exp_ast_l05", "water2", true, true, true, true, "exp"},
   {"086_water_exp_ast_l00", "water2", true, true, true, true, "exp"},
   {"087_water_exp_adt_l10", "water2", true, true, false, true, "exp"},
   {"088_water_exp_adt_l05", "water2", true, true, false, true, "exp"},
   {"089_water_exp_adt_l00", "water2", true, true, false, true, "exp"},
   {"093_water_inv_ast_l10", "water2", true, true, true, true, "inv"},
   {"094_water_inv_ast_l05", "water2", true, true, true, true, "inv"},
   {"095_water_inv_ast_l00", "water2", true, true, true, true, "inv"},
   {"096_water_inv_adt_l10", "water2", true, true, false, true, "inv"},
   {"097_water_inv_adt_l05", "water2", true, true, false, true, "inv"},
   {"098_water_inv_adt_l00", "water2", true, true, false, true, "inv"},
   {"102_water_exf_ast_m10", "water2", true, false, false, true, "exf"},
   {"103_water_exf_ast_m05", "water2", true, false, false, true, "exf"},
   {"104_water_exf_ast_m00", "water2", true, false, false, true, "exf"},
   {"111_water_exf_adt_p10", "water2", false, true, false, true, "exf"},
   {"112_water_exf_adt_p05", "water2", false, true, false, true, "exf"},
   {"113_water_exf_adt_p00", "water2", false, true, false, true, "exf"},
   {"117_water_exf_adt_mp05", "water2", true, true, false, true, "exf"},
   {"119_water_adt_ye_l10", "water2", true, true, false, true, "adt"},
   {"120_water_adt_ne_l10", "water2", true, true, false, true, "adt"},
   {"121_water_adt_ye_l05", "water2", true, true, false, true, "adt"},
   {"122_water_adt_ne_l05", "water2", true, true, false, true, "adt"},
   {"123_water_adt_ye_l00", "water2", true, true, false, true, "adt"},
   {"124_water_adt_ne_l00", "water2", true, true, false, true, "adt"},
   {"131_water_qnt_adt_l10", "water2", true, true, false, true, "legskip"},
   {"132_water_qnt_adt_l00", "water2", true, true, false, true, "legskip"},
   {"135_water_rels_ye_l100", "water2", true, true, true, true, "rels"},
   {"136_water_rels_ye_l085", "water2", true, true, true, true, "rels"},
   {"137_water_rels_ye_l070", "water2", true, true, true, true, "rels"},
   {"138_water_rels_ye_l050", "water2", true, true, true, true, "rels"},
   {"139_water_rels_ye_l030", "water2", true, true, true, true, "rels"},
   {"140_water_rels_ye_l015", "water2", true, true, true, true, "rels"},
   {"141_water_rels_ye_l000", "water2", true, true, true, true, "rels"},
   {"146_water_lmda_ast_l05", "water2", true, true, true, false, "lmda"},
   {"147_water_lmda_ast_e05", "water2", true, true, true, false, "lmda"},
   {"148_water_lmda_ast_l10", "water2", true, true, true, false, "lmda"},
   {"149_water_lmda_ast_none", "water2", true, true, true, false, "lmda"},
   {"150_water_lmda_qnt_l05", "water2", true, true, true, false, "lmda"},
   {"151_water_lmda_vexp_l05", "water2", true, true, true, false, "lmda"},
   {"152_water_lmda_mp05", "water2", true, true, true, false, "lmda"},
   {"153_water_lmda_mp05_expl", "water2", true, true, true, false, "lmda"},
   {"154_water_lmda_e_l06", "water2", true, true, true, false, "lmdadrv"},
   {"155_water_lmda_p_l06", "water2", true, true, true, false, "lmdadrv"},
   {"156_water_lmda_v_l06", "water2", true, true, true, false, "lmdadrv"},
   {"157_water_lmda_ep_l06", "water2", true, true, true, false, "lmdadrv"},
   {"158_water_lmda_ev_l06", "water2", true, true, true, false, "lmdadrv"},
   {"159_water_lmda_pv_l06", "water2", true, true, true, false, "lmdadrv"},
   {"160_water_lmda_epv_l06", "water2", true, true, true, false, "lmdadrv"},
   {"161_water_dlmda_e_l06", "water2", true, true, true, true, "lmdadrv"},
   {"162_water_dlmda_p_l06", "water2", true, true, true, true, "lmdadrv"},
   {"163_water_dlmda_v_l06", "water2", true, true, true, true, "lmdadrv"},
   {"164_water_dlmda_ep_l06", "water2", true, true, true, true, "lmdadrv"},
   {"165_water_dlmda_ev_l06", "water2", true, true, true, true, "lmdadrv"},
   {"166_water_dlmda_pv_l06", "water2", true, true, true, true, "lmdadrv"},
   {"167_water_dlmda_epv_l06", "water2", true, true, true, true, "lmdadrv"},
   {"168_water_rels_ye_vdwm_l030", "water2", true, true, true, true, "rels"},
   {"169_water_rels_ye_lig1_l070", "water2", true, true, true, true, "rels"},
   {"170_water_lmda_ast_epin_l05", "water2", true, true, true, true, "pin"},
   {"171_water_lmda_ast_vpin_l05", "water2", true, true, true, true, "pin"},
   {"173_water_lmda_adt_vpin_l06", "water2", true, true, true, true, "pin"},
   {"176_water_rels_ye_vdwm_exp_l050", "water2", true, true, true, true, "rels"},
   {"177_water_rels_ye_lig2_exp_l030", "water2", true, true, true, true, "rels"},
   {"178_water_rels_ye_lig1_inv_l070", "water2", true, true, true, true, "rels"},
   {"179_water_rels_ye_vdwm_vx3_l050", "water2", true, true, true, true, "rels"},
   {"180_water_rels_ye_lig1_ex3_l085", "water2", true, true, true, true, "rels"},
   {"181_water_rels_ye_lig2_ix2_l015", "water2", true, true, true, true, "rels"},
   {"182_water_ast_v05_annihilate", "water2", true, true, true, true, "ast"},
   {"184_water_exf_adt_l10", "water2", true, true, false, true, "adt"},
   {"185_water_exf_adt_l05", "water2", true, true, false, true, "adt"},
   {"186_water_exf_adt_l00", "water2", true, true, false, true, "adt"},
   {"187_water_ast_ye_l10", "water2", true, true, true, true, "astpol"},
   {"188_water_ast_ne_l10", "water2", true, true, true, true, "astpol"},
   {"189_water_ast_ye_l05", "water2", true, true, true, true, "astpol"},
   {"190_water_ast_ne_l05", "water2", true, true, true, true, "astpol"},
   {"191_water_ast_ye_l00", "water2", true, true, true, true, "astpol"},
   {"192_water_ast_ne_l00", "water2", true, true, true, true, "astpol"},
   {"193_water_exf_ast_l10", "water2", true, true, true, true, "astpol"},
   {"194_water_exf_ast_l05", "water2", true, true, true, true, "astpol"},
   {"195_water_exf_ast_l00", "water2", true, true, true, true, "astpol"},
   {"196_water_apm_ast_vpin_l05", "water2", true, true, true, true, "apm"},
   {"197_water_apm_ast_epin_l00", "water2", true, true, true, true, "apm"},
   {"198_water_apm_ast_epin_l05", "water2", true, true, true, true, "apm"},
   {"199_water_apm_ast_epin_l10", "water2", true, true, true, true, "apm"},
   {"200_water_vsoft_l10", "water2", true, true, true, true, "vsoft"},
   {"201_water_vsoft_l05", "water2", true, true, true, true, "vsoft"},
   {"202_water_vsoft_l00", "water2", true, true, true, true, "vsoft"},
   {"203_water_rels_st_l085", "water2", true, true, true, true, "rels"},
   {"204_water_rels_st_lig2_exp_l030", "water2", true, true, true, true, "rels"},
   // 135 without the LAMBDA keyword: REL-STAGE must default the main lambda to
   // one, so it reproduces the 135 reference.
   {"205_water_rels_nolmda", "water2", true, true, true, true, "rels", "135_water_rels_ye_l100"},
   {"206_trpcage_chiral_m05", "trpcage", false, false, false, false, "chiral"},
   {"207_g3_ast_ye_l10", "g3", true, true, true, true, "g3"},
   {"208_g3_ast_ye_l05", "g3", true, true, true, true, "g3"},
   {"209_g3_ast_annih_l05", "g3", true, true, true, true, "g3"},
   {"210_g3_ast_nobox_l05", "g3", true, true, true, true, "g3"},
   // The long-range van der Waals correction adds a large diagonal virial.
   {"211_water_ast_vcorr_annih_l05", "water2", true, true, true, true, "hal", nullptr, Tols().virial(1.0e-2, 1.0e-2)},
   {"212_water_ast_mono_l05", "water2", true, true, true, true, "hal"},
   {"213_water_ast_tric_l05", "water2", true, true, true, true, "hal"},
   {"214_water_rels_ye_vdwm_lig2t_l040", "water2", true, true, true, true, "hal"},
   // Multipole lambda paths (test_mutate_mpole4): the no-Ewald list path with a
   // 6.5 multipole cutoff, and the charged relative legs, whose uniform background
   // carries a lambda derivative. 216 needs the vacuum Ewald boundary, which
   // tinker9 lacks, so it is left out.
   {"215_water_ast_ne_mcut_l05", "water2", true, true, true, true, "mpole4"},
   {"217_g3_rels_lig1_l085", "g3", true, true, true, true, "mpole4"},
   // Only the water is scaled in this lig2 leg, so d2E/dL2 is what is left of its
   // interaction with itself and its images after the Ewald terms cancel. The
   // single precision reciprocal sum leaves about 1.3e-3 of it behind, which the
   // relative slack of the larger second derivatives elsewhere absorbs; the
   // double precision build reproduces the reference exactly.
   {"218_g3_rels_lig2_l015", "g3", true, true, true, true, "mpole4", nullptr, Tols().lmda2(2.0e-3, 1.0e-4)},
   {"219_water_rels_ne_lig2_l015", "water2", true, true, true, true, "mpole4"},
   // The z-only, 3-fold and z-bisector local frames, with and without Ewald.
   {"220_frames_ast_ye_l05", "frames", true, true, true, true, "frames"},
   {"221_frames_ast_nobox_l05", "frames", true, true, true, true, "frames"},
   // Chignolin and its mirror image, whose chiral frames are inverted under lambda.
   {"222_chig_ast_nobox_l05", "chig", true, true, true, true, "mirror"},
   {"223_chigm_ast_nobox_l05", "chigm", true, true, true, true, "mirror"},
   {"224_chig_ast_ye_l05", "chig", true, true, true, true, "mirror"},
   {"225_chigm_ast_ye_l05", "chigm", true, true, true, true, "mirror"},
   // Earlier cases with LAMBDA-DERIV alone, so polarization takes the single
   // topology dE/dL path (test_mutate_polst). 229 is left out with 216.
   {"226_water_ast_ne_mcut_d1_l05", "water2", true, true, true, true, "polst"},
   {"227_g3_ast_d1_l05", "g3", true, true, true, true, "polst"},
   {"228_g3_ast_d1_l00", "g3", true, true, true, true, "polst"},
   {"230_g3_rels_lig1_d1_l085", "g3", true, true, true, true, "polst"},
   {"231_g3_rels_lig2_d1_l015", "g3", true, true, true, true, "polst"},
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

// The directory holding a base system's coordinates: the SAMPL8 guest 3 has a
// directory of its own, and the water systems sit with the mutation fixtures.
std::string systemDir(const std::string& base)
{
   if (base == "g3")
      return TINKER9_DIRSTR "/test/file/g3/";
   return TINKER9_DIRSTR "/test/file/mutate/";
}

// Copies the parameter files a base system loads into the working directory.
// The water fixtures share water03; the SAMPL8 guest 3 carries its own force
// field, plus the artificial vdw14 values fixture 209 loads as a second file.
// The frames dimer loads amoeba09, and chignolin and its mirror amoebabio09.
std::vector<std::unique_ptr<TestFile>> copyParams(const std::string& base)
{
   std::vector<std::unique_ptr<TestFile>> files;
   if (base == "g3") {
      std::string dir = systemDir(base);
      files.emplace_back(new TestFile(dir + "g3.prm"));
      files.emplace_back(new TestFile(dir + "g3_vdw14.prm"));
   } else if (base == "frames") {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoeba09.prm"));
   } else if (base == "chig" or base == "chigm") {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoebabio09.prm"));
   } else {
      files.emplace_back(new TestFile(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm"));
   }
   return files;
}

// How a run should treat the fused multipole/polarization kernel. emplar cannot
// report interaction counts, so any evaluation that asks for them routes around
// it -- which is why an ordinary run never exercises it at all.
enum class Fuse
{
   Off,     ///< Ordinary run: counts are requested, so emplar is out of reach.
   Require, ///< Drop counts, and require that emplar took over.
   Forbid   ///< Drop counts, and require that it still did not.
};

enum class LmdaMode
{
   Default,
   ThermIntg,
};

void runFixture(const Fixture& fx, Fuse fuse = Fuse::Off, LmdaMode lmdaMode = LmdaMode::Default)
{
   std::string dir = TINKER9_DIRSTR "/test/file/mutate/";
   std::string xyzdst = std::string(fx.base) + ".xyz";
   std::string keyname = std::string(fx.name) + ".key";
   std::string refpath = std::string(TINKER9_DIRSTR "/test/ref/mutate/") + (fx.ref ? fx.ref : fx.name) + ".txt";

   TestFile fxyz(systemDir(fx.base) + xyzdst, xyzdst);
   // TI owns the main lambda and starts at the first schedule window. These
   // four fixtures all reference lambda 0.5, so keep that operating point
   // instead of accepting TI's default first window at lambda 1.
   const char* keyextra = lmdaMode == LmdaMode::ThermIntg ? "\nlambda-mode ti\nti-window 0.5\n" : "";
   TestFile fkey(dir + keyname, keyname, keyextra);
   auto fprm = copyParams(fx.base);

   const char* argv[] = {"dummy", xyzdst.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   const double eps_e = Tols::pick(fx.tol.e, 1.0e-3, 1.0e-4);
   const double eps_g = Tols::pick(fx.tol.g, 1.0e-3, 1.0e-4);
   const double eps_v = Tols::pick(fx.tol.v, 2.0e-3, 1.0e-3);
   // dV/dL is a difference of two endpoint virials of comparable size, so it
   // loses the leading digits the plain virial keeps, and the references print
   // it to three decimals. testlmda.cpp uses the same allowance.
   const double eps_dv = std::max(eps_v, testGetEps(1.0e-2, 2.0e-3));
   const double eps_l = Tols::pick(fx.tol.l, 1.0e-3, 1.0e-4);
   const double eps_l2 = Tols::pick(fx.tol.l2, 1.0e-3, 1.0e-4);

   rc_flag = calc::xyz | calc::mass | calc::vmask;
   if (fuse != Fuse::Off)
      rc_flag &= ~calc::analyz;

   // These key files enable the lambda-derivative machinery through the
   // "lambda-deriv" keyword, so the Fortran-side use_dlmda needs no nudging here.
   testBeginWithArgs(argc, argv);
   initialize();

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

   // Repeat the full check battery twice against the built system.
   for (int irun = 0; irun < 1; ++irun) {
      // v0
      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      // v1
      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      if (fx.dolmda) {
         checkLmdaFirstScalars();
         if (not reducedLmda) {
            checkLmdaSecondScalars();
            checkLmdaGrad();
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
                  COMPARE_REALS(dvirdl[i * 3 + j], lr.dvdl[i][j], eps_dv);
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
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
   }

   finish();
   testEnd();
}

// Runs a fixture twice: once the ordinary way, which asks for interaction
// counts and so goes through the separate empole and epolar kernels, and once
// without counts, where emplar fuses the two. Both are
// checked against the same reference, so the fused kernel has to agree with
// the split one and with Tinker.
void runEmplarFixture(const Fixture& fx)
{
   runFixture(fx);
   runFixture(fx, Fuse::Require);
}

// Reuses a full lambda-scaled fixture with LAMBDA-MODE TI appended to its temporary
// key file. runFixture still exercises v0, v1, v3, v4, v5 and v6; the key keeps
// LAMBDA-DERIV, so the lambda-driven kernels still receive v9 for v1 and v10 for v4.
void runThermIntgFixture(const Fixture& fx)
{
   runFixture(fx, Fuse::Off, LmdaMode::ThermIntg);
}

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

// Runs a fixture whose multipole second lambda derivative is nonzero with and
// without the second, force and virial lambda derivatives (test_mutate_gate).
// Turning them off must leave the energy, the gradient and the first lambda
// derivatives unchanged, and leave the second lambda derivatives at zero.
void runGateFixture(const Fixture& fx)
{
   std::string dir = TINKER9_DIRSTR "/test/file/mutate/";
   std::string xyzdst = std::string(fx.base) + ".xyz";
   std::string keyname = std::string(fx.name) + ".key";

   TestFile fxyz(dir + xyzdst, xyzdst);
   TestFile fkey(dir + keyname, keyname);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzdst.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   const double eps = testGetEps(1.0e-4, 1.0e-8);

   rc_flag = calc::xyz | calc::mass | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   // Full lambda derivatives, as the LAMBDA-DERIV keyword requests.
   REQUIRE(use_d2lmda);
   energy(calc::v4);
   const double e1 = esum, dedl1 = dedl, demdl1 = demdl;
   std::vector<double> gx1(n), gy1(n), gz1(n);
   copyGradient(calc::grad, gx1.data(), gy1.data(), gz1.data());
   REQUIRE(d2emdl2 != 0);

   // Only the first lambda derivative, as TI, META and ABF request.
   use_d2lmda = false;
   REQUIRE(lmdaDerivVers(calc::v4, true) == calc::v8);
   energy(calc::v4);
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
   REQUIRE(d2emdl2 == 0);
   REQUIRE(d2edl2 == 0);
   use_d2lmda = true;

   finish();
   testEnd();
}

// Loads a fixture, with an optional keyword appended to its key file, and
// checks how the derivative keywords choose the polarization topology: the
// first derivative alone keeps single topology, while LAMBDA-DERIV2 needs the
// second derivatives and with them dual topology (test_mutate.f).
void runFlagsFixture(const Fixture& fx, const char* keyextra, bool d2, bool epdt, bool rel)
{
   std::string dir = TINKER9_DIRSTR "/test/file/mutate/";
   std::string xyzdst = std::string(fx.base) + ".xyz";
   std::string keyname = std::string(fx.name) + ".key";

   TestFile fxyz(dir + xyzdst, xyzdst);
   TestFile fkey(dir + keyname, keyname, keyextra);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzdst.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   rc_flag = calc::xyz | calc::mass | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   REQUIRE(use_dlmda);
   REQUIRE(use_d2lmda == d2);
   REQUIRE(use_epdt == epdt);
   REQUIRE((dlmda::use_prst != 0) == not epdt);
   REQUIRE(use_rel == rel);

   finish();
   testEnd();
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
   std::string keyname = std::string(fx.name) + ".key";
   const char* xyzname = "trpcage.xyz";

   TestFile fxyz(TINKER9_DIRSTR "/test/file/trpcage/trpcage.xyz", xyzname);
   TestFile fkey(TINKER9_DIRSTR "/test/file/mutate/" + keyname, keyname,
      "\nlambda-deriv\nlambda 0.5\nvdw-lmda-map exp\n");
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_291a85c1/amoebapro13.prm");

   const char* argv[] = {"dummy", xyzname, "-k", keyname.c_str()};
   int argc = 4;

   const double eps_e = testGetEps(1.0e-3, 1.0e-8);
   const double eps_p = testGetEps(1.0e-6, 1.0e-12);

   rc_flag = calc::xyz | calc::mass | calc::energy;
   testBeginWithArgs(argc, argv);

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

   initialize();
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

   finish();
   testEnd();
}
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
TEST_CASE("MUTATE-047_water_adt_ye_p10", "[ff][mutate][adt]") { runFixture(fx("047_water_adt_ye_p10")); }
TEST_CASE("MUTATE-048_water_adt_ne_p10", "[ff][mutate][adt]") { runFixture(fx("048_water_adt_ne_p10")); }
TEST_CASE("MUTATE-049_water_adt_ye_p05", "[ff][mutate][adt]") { runFixture(fx("049_water_adt_ye_p05")); }
TEST_CASE("MUTATE-050_water_adt_ne_p05", "[ff][mutate][adt]") { runFixture(fx("050_water_adt_ne_p05")); }
TEST_CASE("MUTATE-051_water_adt_ye_p00", "[ff][mutate][adt]") { runFixture(fx("051_water_adt_ye_p00")); }
TEST_CASE("MUTATE-052_water_adt_ne_p00", "[ff][mutate][adt]") { runFixture(fx("052_water_adt_ne_p00")); }
TEST_CASE("MUTATE-056_water_adt_ye_mp05", "[ff][mutate][adt]") { runFixture(fx("056_water_adt_ye_mp05")); }
TEST_CASE("MUTATE-057_water_adt_ne_mp05", "[ff][mutate][adt]") { runFixture(fx("057_water_adt_ne_mp05")); }
TEST_CASE("MUTATE-075_water_qnt_ast_l10", "[ff][mutate][qnt]") { runFixture(fx("075_water_qnt_ast_l10")); }
TEST_CASE("MUTATE-076_water_qnt_ast_l05", "[ff][mutate][qnt]") { runFixture(fx("076_water_qnt_ast_l05")); }
TEST_CASE("MUTATE-077_water_qnt_ast_l00", "[ff][mutate][qnt]") { runFixture(fx("077_water_qnt_ast_l00")); }
TEST_CASE("MUTATE-078_water_qnt_adt_l10", "[ff][mutate][qnt]") { runFixture(fx("078_water_qnt_adt_l10")); }
TEST_CASE("MUTATE-079_water_qnt_adt_l05", "[ff][mutate][qnt]") { runFixture(fx("079_water_qnt_adt_l05")); }
TEST_CASE("MUTATE-080_water_qnt_adt_l00", "[ff][mutate][qnt]") { runFixture(fx("080_water_qnt_adt_l00")); }
TEST_CASE("MUTATE-084_water_exp_ast_l10", "[ff][mutate][exp]") { runFixture(fx("084_water_exp_ast_l10")); }
TEST_CASE("MUTATE-085_water_exp_ast_l05", "[ff][mutate][exp]") { runFixture(fx("085_water_exp_ast_l05")); }
TEST_CASE("MUTATE-086_water_exp_ast_l00", "[ff][mutate][exp]") { runFixture(fx("086_water_exp_ast_l00")); }
TEST_CASE("MUTATE-087_water_exp_adt_l10", "[ff][mutate][exp]") { runFixture(fx("087_water_exp_adt_l10")); }
TEST_CASE("MUTATE-088_water_exp_adt_l05", "[ff][mutate][exp]") { runFixture(fx("088_water_exp_adt_l05")); }
TEST_CASE("MUTATE-089_water_exp_adt_l00", "[ff][mutate][exp]") { runFixture(fx("089_water_exp_adt_l00")); }
TEST_CASE("MUTATE-093_water_inv_ast_l10", "[ff][mutate][inv]") { runFixture(fx("093_water_inv_ast_l10")); }
TEST_CASE("MUTATE-094_water_inv_ast_l05", "[ff][mutate][inv]") { runFixture(fx("094_water_inv_ast_l05")); }
TEST_CASE("MUTATE-095_water_inv_ast_l00", "[ff][mutate][inv]") { runFixture(fx("095_water_inv_ast_l00")); }
TEST_CASE("MUTATE-096_water_inv_adt_l10", "[ff][mutate][inv]") { runFixture(fx("096_water_inv_adt_l10")); }
TEST_CASE("MUTATE-097_water_inv_adt_l05", "[ff][mutate][inv]") { runFixture(fx("097_water_inv_adt_l05")); }
TEST_CASE("MUTATE-098_water_inv_adt_l00", "[ff][mutate][inv]") { runFixture(fx("098_water_inv_adt_l00")); }
TEST_CASE("MUTATE-102_water_exf_ast_m10", "[ff][mutate][exf]") { runFixture(fx("102_water_exf_ast_m10")); }
TEST_CASE("MUTATE-103_water_exf_ast_m05", "[ff][mutate][exf]") { runFixture(fx("103_water_exf_ast_m05")); }
TEST_CASE("MUTATE-104_water_exf_ast_m00", "[ff][mutate][exf]") { runFixture(fx("104_water_exf_ast_m00")); }
TEST_CASE("MUTATE-111_water_exf_adt_p10", "[ff][mutate][exf]") { runFixture(fx("111_water_exf_adt_p10")); }
TEST_CASE("MUTATE-112_water_exf_adt_p05", "[ff][mutate][exf]") { runFixture(fx("112_water_exf_adt_p05")); }
TEST_CASE("MUTATE-113_water_exf_adt_p00", "[ff][mutate][exf]") { runFixture(fx("113_water_exf_adt_p00")); }
TEST_CASE("MUTATE-117_water_exf_adt_mp05", "[ff][mutate][exf]") { runFixture(fx("117_water_exf_adt_mp05")); }
TEST_CASE("MUTATE-119_water_adt_ye_l10", "[ff][mutate][adt]") { runFixture(fx("119_water_adt_ye_l10")); }
TEST_CASE("MUTATE-120_water_adt_ne_l10", "[ff][mutate][adt]") { runFixture(fx("120_water_adt_ne_l10")); }
TEST_CASE("MUTATE-121_water_adt_ye_l05", "[ff][mutate][adt]") { runFixture(fx("121_water_adt_ye_l05")); }
TEST_CASE("MUTATE-122_water_adt_ne_l05", "[ff][mutate][adt]") { runFixture(fx("122_water_adt_ne_l05")); }
TEST_CASE("MUTATE-123_water_adt_ye_l00", "[ff][mutate][adt]") { runFixture(fx("123_water_adt_ye_l00")); }
TEST_CASE("MUTATE-124_water_adt_ne_l00", "[ff][mutate][adt]") { runFixture(fx("124_water_adt_ne_l00")); }
TEST_CASE("MUTATE-131_water_qnt_adt_l10", "[ff][mutate][legskip]") { runLegSkipFixture(fx("131_water_qnt_adt_l10"), false, true); }
TEST_CASE("MUTATE-132_water_qnt_adt_l00", "[ff][mutate][legskip]") { runLegSkipFixture(fx("132_water_qnt_adt_l00"), true, false); }
TEST_CASE("MUTATE-135_water_rels_ye_l100", "[ff][mutate][rels]") { runFixture(fx("135_water_rels_ye_l100")); }
TEST_CASE("MUTATE-136_water_rels_ye_l085", "[ff][mutate][rels]") { runFixture(fx("136_water_rels_ye_l085")); }
TEST_CASE("MUTATE-137_water_rels_ye_l070", "[ff][mutate][rels]") { runFixture(fx("137_water_rels_ye_l070")); }
TEST_CASE("MUTATE-138_water_rels_ye_l050", "[ff][mutate][rels]") { runFixture(fx("138_water_rels_ye_l050")); }
TEST_CASE("MUTATE-139_water_rels_ye_l030", "[ff][mutate][rels]") { runFixture(fx("139_water_rels_ye_l030")); }
TEST_CASE("MUTATE-140_water_rels_ye_l015", "[ff][mutate][rels]") { runFixture(fx("140_water_rels_ye_l015")); }
TEST_CASE("MUTATE-141_water_rels_ye_l000", "[ff][mutate][rels]") { runFixture(fx("141_water_rels_ye_l000")); }
TEST_CASE("MUTATE-146_water_lmda_ast_l05", "[ff][mutate][lmda]") { runFixture(fx("146_water_lmda_ast_l05")); }
TEST_CASE("MUTATE-147_water_lmda_ast_e05", "[ff][mutate][lmda]") { runFixture(fx("147_water_lmda_ast_e05")); }
TEST_CASE("MUTATE-148_water_lmda_ast_l10", "[ff][mutate][lmda]") { runFixture(fx("148_water_lmda_ast_l10")); }
TEST_CASE("MUTATE-149_water_lmda_ast_none", "[ff][mutate][lmda]") { runFixture(fx("149_water_lmda_ast_none")); }
TEST_CASE("MUTATE-150_water_lmda_qnt_l05", "[ff][mutate][lmda]") { runFixture(fx("150_water_lmda_qnt_l05")); }
TEST_CASE("MUTATE-151_water_lmda_vexp_l05", "[ff][mutate][lmda]") { runFixture(fx("151_water_lmda_vexp_l05")); }
TEST_CASE("MUTATE-152_water_lmda_mp05", "[ff][mutate][lmda]") { runFixture(fx("152_water_lmda_mp05")); }
TEST_CASE("MUTATE-153_water_lmda_mp05_expl", "[ff][mutate][lmda]") { runFixture(fx("153_water_lmda_mp05_expl")); }
TEST_CASE("MUTATE-154_water_lmda_e_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("154_water_lmda_e_l06")); }
TEST_CASE("MUTATE-155_water_lmda_p_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("155_water_lmda_p_l06")); }
TEST_CASE("MUTATE-156_water_lmda_v_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("156_water_lmda_v_l06")); }
TEST_CASE("MUTATE-157_water_lmda_ep_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("157_water_lmda_ep_l06")); }
TEST_CASE("MUTATE-158_water_lmda_ev_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("158_water_lmda_ev_l06")); }
TEST_CASE("MUTATE-159_water_lmda_pv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("159_water_lmda_pv_l06")); }
TEST_CASE("MUTATE-160_water_lmda_epv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("160_water_lmda_epv_l06")); }
TEST_CASE("MUTATE-161_water_dlmda_e_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("161_water_dlmda_e_l06")); }
TEST_CASE("MUTATE-162_water_dlmda_p_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("162_water_dlmda_p_l06")); }
TEST_CASE("MUTATE-163_water_dlmda_v_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("163_water_dlmda_v_l06")); }
TEST_CASE("MUTATE-164_water_dlmda_ep_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("164_water_dlmda_ep_l06")); }
TEST_CASE("MUTATE-165_water_dlmda_ev_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("165_water_dlmda_ev_l06")); }
TEST_CASE("MUTATE-166_water_dlmda_pv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("166_water_dlmda_pv_l06")); }
TEST_CASE("MUTATE-167_water_dlmda_epv_l06", "[ff][mutate][lmdadrv]") { runFixture(fx("167_water_dlmda_epv_l06")); }
TEST_CASE("MUTATE-168_water_rels_ye_vdwm_l030", "[ff][mutate][rels]") { runFixture(fx("168_water_rels_ye_vdwm_l030")); }
TEST_CASE("MUTATE-169_water_rels_ye_lig1_l070", "[ff][mutate][rels]") { runFixture(fx("169_water_rels_ye_lig1_l070")); }
TEST_CASE("MUTATE-170_water_lmda_ast_epin_l05", "[ff][mutate][pin]") { runFixture(fx("170_water_lmda_ast_epin_l05")); }
TEST_CASE("MUTATE-171_water_lmda_ast_vpin_l05", "[ff][mutate][pin]") { runFixture(fx("171_water_lmda_ast_vpin_l05")); }
TEST_CASE("MUTATE-173_water_lmda_adt_vpin_l06", "[ff][mutate][pin]") { runFixture(fx("173_water_lmda_adt_vpin_l06")); }
TEST_CASE("MUTATE-176_water_rels_ye_vdwm_exp_l050", "[ff][mutate][rels]") { runFixture(fx("176_water_rels_ye_vdwm_exp_l050")); }
TEST_CASE("MUTATE-177_water_rels_ye_lig2_exp_l030", "[ff][mutate][rels]") { runFixture(fx("177_water_rels_ye_lig2_exp_l030")); }
TEST_CASE("MUTATE-178_water_rels_ye_lig1_inv_l070", "[ff][mutate][rels]") { runFixture(fx("178_water_rels_ye_lig1_inv_l070")); }
TEST_CASE("MUTATE-179_water_rels_ye_vdwm_vx3_l050", "[ff][mutate][rels]") { runFixture(fx("179_water_rels_ye_vdwm_vx3_l050")); }
TEST_CASE("MUTATE-180_water_rels_ye_lig1_ex3_l085", "[ff][mutate][rels]") { runFixture(fx("180_water_rels_ye_lig1_ex3_l085")); }
TEST_CASE("MUTATE-181_water_rels_ye_lig2_ix2_l015", "[ff][mutate][rels]") { runFixture(fx("181_water_rels_ye_lig2_ix2_l015")); }
TEST_CASE("MUTATE-182_water_ast_v05_annihilate", "[ff][mutate][ast]") { runFixture(fx("182_water_ast_v05_annihilate")); }
TEST_CASE("MUTATE-184_water_exf_adt_l10", "[ff][mutate][exf]") { runFixture(fx("184_water_exf_adt_l10")); }
TEST_CASE("MUTATE-185_water_exf_adt_l05", "[ff][mutate][exf]") { runFixture(fx("185_water_exf_adt_l05")); }
TEST_CASE("MUTATE-186_water_exf_adt_l00", "[ff][mutate][exf]") { runFixture(fx("186_water_exf_adt_l00")); }
TEST_CASE("MUTATE-187_water_ast_ye_l10", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("187_water_ast_ye_l10")); }
TEST_CASE("MUTATE-188_water_ast_ne_l10", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("188_water_ast_ne_l10")); }
TEST_CASE("MUTATE-189_water_ast_ye_l05", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("189_water_ast_ye_l05")); }
TEST_CASE("MUTATE-190_water_ast_ne_l05", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("190_water_ast_ne_l05")); }
TEST_CASE("MUTATE-191_water_ast_ye_l00", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("191_water_ast_ye_l00")); }
TEST_CASE("MUTATE-192_water_ast_ne_l00", "[ff][mutate][ast][astpol][emplar]") { runEmplarAstFixture(fx("192_water_ast_ne_l00")); }
TEST_CASE("MUTATE-193_water_exf_ast_l10", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("193_water_exf_ast_l10")); }
TEST_CASE("MUTATE-194_water_exf_ast_l05", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("194_water_exf_ast_l05")); }
TEST_CASE("MUTATE-195_water_exf_ast_l00", "[ff][mutate][exf][ast][astpol][emplar]") { runEmplarAstFixture(fx("195_water_exf_ast_l00")); }
TEST_CASE("MUTATE-196_water_apm_ast_vpin_l05", "[ff][mutate][apm][pin]") { runFixture(fx("196_water_apm_ast_vpin_l05")); }
TEST_CASE("MUTATE-197_water_apm_ast_epin_l00", "[ff][mutate][apm][pin]") { runFixture(fx("197_water_apm_ast_epin_l00")); }
TEST_CASE("MUTATE-198_water_apm_ast_epin_l05", "[ff][mutate][apm][pin]") { runFixture(fx("198_water_apm_ast_epin_l05")); }
TEST_CASE("MUTATE-199_water_apm_ast_epin_l10", "[ff][mutate][apm][pin]") { runFixture(fx("199_water_apm_ast_epin_l10")); }
TEST_CASE("MUTATE-200_water_vsoft_l10", "[ff][mutate][vsoft]") { runFixture(fx("200_water_vsoft_l10")); }
TEST_CASE("MUTATE-201_water_vsoft_l05", "[ff][mutate][vsoft]") { runFixture(fx("201_water_vsoft_l05")); }
TEST_CASE("MUTATE-202_water_vsoft_l00", "[ff][mutate][vsoft]") { runFixture(fx("202_water_vsoft_l00")); }
TEST_CASE("MUTATE-203_water_rels_st_l085", "[ff][mutate][rels][astpol][emplar]") { runEmplarAstFixture(fx("203_water_rels_st_l085")); }
TEST_CASE("MUTATE-204_water_rels_st_lig2_exp_l030", "[ff][mutate][rels][astpol][emplar]") { runEmplarAstFixture(fx("204_water_rels_st_lig2_exp_l030")); }
TEST_CASE("MUTATE-205_water_rels_nolmda", "[ff][mutate][rels]") { runFixture(fx("205_water_rels_nolmda")); }
TEST_CASE("MUTATE-207_g3_ast_ye_l10", "[ff][mutate][g3]") { runFixture(fx("207_g3_ast_ye_l10")); }
TEST_CASE("MUTATE-208_g3_ast_ye_l05", "[ff][mutate][g3]") { runFixture(fx("208_g3_ast_ye_l05")); }
TEST_CASE("MUTATE-209_g3_ast_annih_l05", "[ff][mutate][g3]") { runFixture(fx("209_g3_ast_annih_l05")); }
TEST_CASE("MUTATE-210_g3_ast_nobox_l05", "[ff][mutate][g3]") { runFixture(fx("210_g3_ast_nobox_l05")); }
TEST_CASE("MUTATE-211_water_ast_vcorr_annih_l05", "[ff][mutate][hal]") { runFixture(fx("211_water_ast_vcorr_annih_l05")); }
TEST_CASE("MUTATE-212_water_ast_mono_l05", "[ff][mutate][hal]") { runFixture(fx("212_water_ast_mono_l05")); }
TEST_CASE("MUTATE-213_water_ast_tric_l05", "[ff][mutate][hal]") { runFixture(fx("213_water_ast_tric_l05")); }
TEST_CASE("MUTATE-214_water_rels_ye_vdwm_lig2t_l040", "[ff][mutate][hal]") { runFixture(fx("214_water_rels_ye_vdwm_lig2t_l040")); }
TEST_CASE("MUTATE-215_water_ast_ne_mcut_l05", "[ff][mutate][mpole4]") { runFixture(fx("215_water_ast_ne_mcut_l05")); }
TEST_CASE("MUTATE-217_g3_rels_lig1_l085", "[ff][mutate][mpole4]") { runFixture(fx("217_g3_rels_lig1_l085")); }
TEST_CASE("MUTATE-218_g3_rels_lig2_l015", "[ff][mutate][mpole4]") { runFixture(fx("218_g3_rels_lig2_l015")); }
TEST_CASE("MUTATE-219_water_rels_ne_lig2_l015", "[ff][mutate][mpole4]") { runFixture(fx("219_water_rels_ne_lig2_l015")); }
TEST_CASE("MUTATE-220_frames_ast_ye_l05", "[ff][mutate][frames]") { runFixture(fx("220_frames_ast_ye_l05")); }
TEST_CASE("MUTATE-221_frames_ast_nobox_l05", "[ff][mutate][frames]") { runFixture(fx("221_frames_ast_nobox_l05")); }
TEST_CASE("MUTATE-222_chig_ast_nobox_l05", "[ff][mutate][mirror]") { runFixture(fx("222_chig_ast_nobox_l05")); }
TEST_CASE("MUTATE-223_chigm_ast_nobox_l05", "[ff][mutate][mirror]") { runFixture(fx("223_chigm_ast_nobox_l05")); }
TEST_CASE("MUTATE-224_chig_ast_ye_l05", "[ff][mutate][mirror]") { runFixture(fx("224_chig_ast_ye_l05")); }
TEST_CASE("MUTATE-225_chigm_ast_ye_l05", "[ff][mutate][mirror]") { runFixture(fx("225_chigm_ast_ye_l05")); }
TEST_CASE("MUTATE-226_water_ast_ne_mcut_d1_l05", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("226_water_ast_ne_mcut_d1_l05")); }
TEST_CASE("MUTATE-227_g3_ast_d1_l05", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("227_g3_ast_d1_l05")); }
TEST_CASE("MUTATE-228_g3_ast_d1_l00", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("228_g3_ast_d1_l00")); }
TEST_CASE("MUTATE-230_g3_rels_lig1_d1_l085", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("230_g3_rels_lig1_d1_l085")); }
TEST_CASE("MUTATE-231_g3_rels_lig2_d1_l015", "[ff][mutate][polst][astpol][emplar]") { runEmplarAstFixture(fx("231_g3_rels_lig2_d1_l015")); }

TEST_CASE("MUTATE-TI-076_water_qnt_ast_l05", "[ff][mutate][ti][ast]") { runThermIntgFixture(fx("076_water_qnt_ast_l05")); }
TEST_CASE("MUTATE-TI-079_water_qnt_adt_l05", "[ff][mutate][ti][adt]") { runThermIntgFixture(fx("079_water_qnt_adt_l05")); }
TEST_CASE("MUTATE-TI-176_water_rels_ye_vdwm_exp_l050", "[ff][mutate][ti][rels]") {runThermIntgFixture(fx("176_water_rels_ye_vdwm_exp_l050"));}

// The fused kernel on its own, for fixtures with no lambda derivative and one
// lambda value for electrostatics and polarization, so plain emplar takes over.
TEST_CASE("MUTATE-EMPLAR-001_water_ye_m10", "[ff][mutate][emplar]") { runFixture(fx("001_water_ye_m10"), Fuse::Require); }
TEST_CASE("MUTATE-EMPLAR-003_water_ye_m05", "[ff][mutate][emplar]") { runFixture(fx("003_water_ye_m05"), Fuse::Require); }
TEST_CASE("MUTATE-EMPLAR-147_water_lmda_ast_e05", "[ff][mutate][emplar]") { runFixture(fx("147_water_lmda_ast_e05"), Fuse::Require); }

TEST_CASE("MUTATE-206_trpcage_chiral_m05", "[ff][mutate][chiral]") { runChiralFixture(fx("206_trpcage_chiral_m05")); }

TEST_CASE("MUTATE-gate", "[ff][mutate][rels]") { runGateFixture(fx("136_water_rels_ye_l085")); }

TEST_CASE("MUTATE-flags", "[ff][mutate][rels]")
{
   runFlagsFixture(fx("152_water_lmda_mp05"), "\nLAMBDA-DERIV\n", false, false, false);
   runFlagsFixture(fx("152_water_lmda_mp05"), "\nLAMBDA-DERIV2\n", true, true, false);
   runFlagsFixture(fx("203_water_rels_st_l085"), "", false, false, true);
   runFlagsFixture(fx("136_water_rels_ye_l085"), "", true, true, true);
}

#endif
