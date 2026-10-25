#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <cmath>
#include <fstream>
#include <string>
#include <tinker/detail/boxes.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/routines.h>

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
struct Fixture
{
   const char* name;
   double eps = 1.0e-2;  ///< Finite difference stepsize, in lambda.
   double ntol = 1.0;    ///< Scale on the tolerances of the numerical derivatives.
   double dtol = 1.0e-4; ///< Tolerance of the analytical derivatives in double precision.
   const char* xyz = "testlmda/water2.xyz";         ///< Coordinates, under test/file.
   const char* prm = "commit_6fe8e913/water03.prm"; ///< Parameters, under test/file.
   /// Tolerance of the analytical second derivative in mixed precision. A
   /// steep staged map squares its slope into d2E/dL2, and large terms that
   /// cancel leave it 1e-3 to 2e-3 off there; a double build reproduces it.
   double d2tol = 1.0e-3;
};

const Fixture kFixtures[] = {
   {"01_water_adt_l05"},
   {"02_water_ast_l05"},
   {"03_water_adt_l06exp"},
   {"04_water_ast_l06exp"},
   {"05_water_ast_nodl_l05"},
   {"06_water_ast_vonly_l05"},
   {"07_water_ast_vcorr_l05"},
   {"08_water_ast_vcorr_annih_l05"},
   {"09_water_ast_vcorr_l06exp"},
   {"10_water_rels_vdwm_vcorr_annih_l05"},
   {"11_water_rels_vdwm_vcorr_l050", 2.0e-3, 2.0},
   {"12_water_rels_lig1_l085", 4.0e-3, 3.0},
   {"13_water_rels_lig2_l015", 4.0e-3, 3.0, 5.0e-4},
   {"14_water_rels_lig1_ne_l085", 4.0e-3, 3.0},
   // the external field makes E_pol -134 kcal/mol, whose float-atomic noise in
   // mixed precision moves the finite difference d2E/dL2 by 0.49 +/- 0.10
   {"15_water_rels_lig1_nlist_exf_l085", 4.0e-3, 6.0},
   {"16_water_rels_lig1_st_l085", 4.0e-3, 3.0},
   {"17_water_rels_lig2_st_l015", 4.0e-3, 3.0},
   {"18_water_rels_lig1_st_ne_l085", 4.0e-3, 3.0},
   {"19_water_rels_lig1_st_polonly_l085", 4.0e-3, 3.0},
   {"20_water_rels_lig1_dt_polonly_l078", 4.0e-3, 3.0},
   {"21_water_rels_lig1_st_l088", 4.0e-3, 3.0},
   {"22_water_rels_lig1_dt_l088", 4.0e-3, 3.0},
   {"23_water_rels_3stg_l085", 4.0e-3, 3.0},
   {"24_ionwat_ewald_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"25_ionwat_ewald_nlist_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"26_ionwat_pol_ewald_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"27_water_rels_2stg_l075", 4.0e-3, 3.0},
   // charge penetration multipoles, with charge flux but for 40 (test_testlmda.f)
   {"28_hipwat_flux_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"29_hipwat_flux_ewald_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"30_hipwat_flux_ewald_nlist_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"31_hipwat_flux_ewald_lin_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"32_aptetra_flux_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"33_aptetra_flux_ewald_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"34_aptetra_flux_ewald_nlist_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"35_aptetra_flux_ewald_lin_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"36_apgas_flux_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"37_apgas_cross_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"38_apgas_cross_lin_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"39_apgas_cross_ewald_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"40_hipmix_ewald_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"58_hipwat_flux_mlist_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   // polarization with charge penetration, in single topology
   {"41_hipwat_pol_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"42_hipwat_pol_ewald_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"43_hipwat_pol_ewald_nlist_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"44_hipwat_pol_ewald_lin_st_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"46_aptetra_pol_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"47_aptetra_pol_ewald_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"48_aptetra_pol_ewald_nlist_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"49_aptetra_pol_ewald_lin_st_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"51_apgas_pol_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"53_hipmix_pol_ewald_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"57_hipwat_pol_split_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   // in dual topology, and relative legs; the AMOEBA+ legs take their
   // derivatives across a 0.3 wide stage, whose slope carries the polar-eps
   // convergence of each endpoint's dipoles into dF/dL at about 4e-4
   {"45_hipwat_pol_ewald_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"50_aptetra_pol_ewald_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"52_apgas_pol_ewald_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"54_hipmix_pol_ewald_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"56_hipwat_pol_split_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm", 4.0e-3},
   {"59_aptetra_pol_mlist_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"60_aptetra_rel_lig1_l085", 4.0e-3, 3.0, 6.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm",
      4.0e-3},
   {"61_aptetra_rel_lig2_l015", 4.0e-3, 3.0, 6.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm",
      4.0e-3},
   // the multipole brick spread makes the 39-atom grid sum order vary between
   // runs, which 4/eps^2 turns into 0.16 sd of the finite difference d2E/dL2
   // at a step of 4e-3; this step and tolerance keep it and the truncation of
   // dE/dL well inside their margins
   {"62_aptetra_rel_cross_l085", 6.0e-3, 5.0, 6.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"63_hipwat_rel_lig1_ewald_l085", 4.0e-3, 3.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm", 4.0e-3},
   // HIPPO polarization driven alone
   {"64_hipwat_pol_only_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"65_hipwat_pol_only_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   // HIPPO polarization with exchange polarization
   {"66_expnaclus_na_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm"},
   {"67_expnaclus_wat_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm"},
   // the finite difference d2E/dL2 (-216) takes the round-off of the multipole
   // energy times 4/eps^2 in mixed precision, about 3x more with fast math in a
   // Release build, and dE/dL takes truncation above; this step keeps both at
   // about a third of their tolerances in Debug and Release
   {"68_expbox_ewald_dt_l04", 1.25e-2, 3.0, 1.0e-4, "testlmda/expbox.xyz", "commit_dd1761ac/expol.prm"},
   {"69_expnaclus_wat_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm"},
   {"70_expbox_ewald_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/expbox.xyz", "commit_dd1761ac/expol.prm"},
   {"71_expnaclus_na_lin_st_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm"},
   {"72_expbox_ewald_nlist_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/expbox.xyz", "commit_dd1761ac/expol.prm"},
   // charge transfer follows the polarization lambda
   {"73_hipwat_ct_only_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"74_hipwat_ct_only_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"75_hipwat_ct_only_dtexp_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"76_hipwat_ct_mlist_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"77_hipwat_ct_lin_dt_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"78_hipwat_pol_ct_split_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"79_hipwat_pol_ct_ewald_dt_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"80_aptetra_ct_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"82_hipwat_ct_group_ost_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"83_hipwat_ct_off_st_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"84_aptetra_ct_rel_lig1_st_l085", 4.0e-3, 3.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"85_aptetra_ct_rel_lig2_st_l015", 4.0e-3, 3.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"86_aptetra_ct_rel_lig1_dt_l080", 4.0e-3, 3.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"87_aptetra_ct_rel_lig2_dt_l020", 4.0e-3, 3.0, 1.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"88_aptetra_ct_rel_vdwm_l050", 4.0e-3, 3.0, 1.0e-4, "testlmda/aptetra.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   // the stage slope carries polar-eps convergence into dF/dL at about 4e-4, as
   // in 60-62, and its finite difference d2E/dL2 (-1652) is 4e-4 off at this step
   {"89_aptetra_pol_ct_rel_lig1_ewald_l080", 4.0e-3, 4.0, 6.0e-4, "testlmda/aptetra.xyz",
      "commit_152ecec0/AMOEBAplus_Org.prm"},
   // d2E/dL2 is 1.4e-3 off in mixed precision; a double build reproduces it
   {"90_hipwat_pol_ct_rel_lig1_ewald_l080", 4.0e-3, 3.0, 1.0e-4, "testlmda/hipwat.xyz", "commit_aad9340c/water21.prm",
      4.0e-3},
   {"91_hipmix_rep_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"92_hipmix_rep_nlist_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"93_hipmix_rep_annih_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"94_hipmix_rep_taper_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"95_hipmix_rep_sc_m0p2_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"96_hipmix_rep_sc_real_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"97_hipmix_rep_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"98_hipmix_rep_l10", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"99_hipmix_rep_sc_m0p2_l10", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"100_hipmix_rep_d1_n12_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"117_hipmix_rep_exp3_l08", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"101_hipmix_dsp_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"102_hipmix_dsp_nlist_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"103_hipmix_dsp_annih_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"104_hipmix_dsp_taper_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"105_hipmix_dsp_ewald_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"106_hipmix_dsp_ewald_nlist_annih_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"107_hipmix_dsp_sc_real_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"108_hipmix_dsp_ewald_n2_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"109_hipmix_dsp_ewald_l10", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"110_hipmix_dsp_d1_n1_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"112_hipmix_dsp_ewald_group_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"113_hipmix_dsp_corr_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"114_hipmix_dsp_corr_annih_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"115_hipmix_dsp_corr_n1_l04", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"116_hipmix_dsp_corr_n2_l00", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"118_water_vsoft_m15_p3_l05"},
   {"119_water_vsoft_m05_p1_l03"},
   {"120_hipmix_dsp_ewald_exp3_l08", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"121_hipmix_dsp_corr_exp3_l06", 1.0e-2, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"122_hipmix_rep_qnt_l045", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"123_hipmix_rep_inv_l03", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"124_hipmix_dsp_corr_qnt_l06", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   {"125_hipmix_dsp_ewald_inv_l04", 5.0e-3, 1.0, 1.0e-4, "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
};

// The fixture of a given name. Cases look their fixture up by name so that
// retiring a fixture does not shift every case after it.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no testlmda fixture named " << name);
   return kFixtures[0];
}

// With analyz off the fixture runs as dynamics does: the terms keep no lambda
// derivative buffers of their own and add into the shared ones, so only the
// totals are reported and compared.
void runFixture(const Fixture& fx, bool analyz = true)
{
   // The lambda fixtures leave the derivative machinery switched on; clear it
   // once the case ends, however it ends.
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);

   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testlmda/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   TestSession session(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = fx.eps;

   rc_flag = testlmdaFlags(opts);
   if (not analyz)
      rc_flag &= ~calc::analyz;
   session.init();
   if (not analyz) {
      auto shared = [](EnergyBuffer term, EnergyBuffer total) { return term == nullptr or term == total; };
      if (use_dlmda)
         REQUIRE(dedl_buf != nullptr);
      REQUIRE(shared(demdl_buf, dedl_buf));
      REQUIRE(shared(depdl_buf, dedl_buf));
      REQUIRE(shared(devdl_buf, dedl_buf));
      REQUIRE(shared(d2emdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2epdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2evdl2_buf, d2edl2_buf));
   }
   // The total and, under analysis, the van der Waals, multipole and
   // polarization breakdown.
   const int nterm = analyz ? 4 : 1;

   auto r = testlmdaEvaluate(opts);
   TestReference reffile(std::string(TINKER9_DIRSTR "/test/ref/testlmda/") + fx.name + ".txt");
   const TestLmdaReference& ref = reffile.getLmda();
   REQUIRE((int)ref.lgrad.size() >= n);

   const double eps_d = testGetEps(1.0e-3, fx.dtol);
   const double eps_d2 = testGetEps(fx.d2tol, fx.dtol);
   // dV/dL is printed with 3 decimals in the references.
   const double eps_v = 1.0e-2;

   // ---- Analytical values against the reference ----------------------------
   for (int k = 0; k < nterm; ++k) {
      COMPARE_REALS(r.dedl[k], ref.dedl[k], eps_d);
      COMPARE_REALS(r.d2edl2[k], ref.d2edl2[k], eps_d2);
   }
   COMPARE_GRADIENT_FLAT(r.dfdl, ref.lgrad, eps_d);
   COMPARE_VIR9(r.dvirdl, ref.dvdl, eps_v);

   // ---- Numerical values against the same reference ------------------------
   // Central-difference truncation dominates here; the second derivatives carry
   // an extra factor of 1/eps of it.
   const double eps_n1 = 5.0e-2 * fx.ntol;
   const double eps_n2 = 2.0e-1 * fx.ntol;
   const double eps_nf = 1.0e-1 * fx.ntol;
   for (int k = 0; k < nterm; ++k) {
      COMPARE_REALS(r.ndedl[k], ref.dedl[k], eps_n1);
      COMPARE_REALS(r.nd2edl2[k], ref.d2edl2[k], eps_n2);
   }
   COMPARE_GRADIENT_FLAT(r.ndfdl, ref.lgrad, eps_nf);
   COMPARE_VIR9(r.ndvirdl, ref.dvdl, eps_nf);

   session.end();
}

// The analytical lambda derivatives of a fixture from two energy calls in a row
// agree, as a dynamics run takes them step after step.
void runRepeatedFixture(const Fixture& fx)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);
   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testlmda/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   rc_flag = testlmdaFlags(opts);
   session.init();
   energy(calc::v1);
   const double dedl1 = dedl, depdl1 = depdl;
   energy(calc::v1);
   const double eps = testGetEps(1.0e-5, 1.0e-10);
   COMPARE_REALS(dedl, dedl1, eps);
   COMPARE_REALS(depdl, depdl1, eps);
   session.end();
}

// A fixture whose key tinker9 refuses as it sets up. The throw comes from
// elecData(), before pmeData() opens anything a later case could trip on.
void runRefusedFixture(const Fixture& fx)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);
   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testlmda/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   rc_flag = testlmdaFlags(opts);
   REQUIRE_THROWS(session.init());
}

// The long range correction from its class sums against Fortran evcorr1 at a
// few van der Waals lambdas, for the van der Waals (dsp false) or the
// dispersion correction. Fortran takes the second lambda derivative only with
// use_d2lmda, and its virial is minus the energy. evcorr1 compares its whole
// character*6 mode, so the mode is padded to six characters.
void runLrcFixture(const Fixture& fx, bool dsp)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);
   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testlmda/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass | calc::energy;
   session.init();

   LrcSums ls;
   lrcSetup(ls, dsp);
   REQUIRE(ls.s0 != 0);
   REQUIRE(ls.s1 != 0);
   const double vol = boxes::volbox;
   const double saved = mutant::vlambda;
   for (double vl : {0.0, 0.3, 0.7, 1.0}) {
      INFO("vlambda " << vl);
      mutant::vlambda = vl;
      double elrc, vlrc, delrc, dvlrc, d2elrc;
      if (dsp)
         tinker_f_evcorr1({const_cast<char*>("DISP  "), 6}, &elrc, &vlrc, &delrc, &dvlrc, &d2elrc);
      else
         tinker_f_evcorr1({const_cast<char*>("VDW   "), 6}, &elrc, &vlrc, &delrc, &dvlrc, &d2elrc);
      // with respect to vlambda itself
      const LmdaFn f = lrcEval(ls, vl, 1, 0);
      const double e = f.f, de = f.d1, d2e = f.d2;
      auto near = [](double a, double b) { return std::fabs(a - b) <= 1.0e-12 * std::fabs(b) + 1.0e-15; };
      CAPTURE(fx.name, e / vol, elrc, de / vol, delrc, d2e / vol, d2elrc);
      REQUIRE(near(e / vol, elrc));
      REQUIRE(near(de / vol, delrc));
      REQUIRE(near(d2e / vol, d2elrc));
      REQUIRE(vlrc == -elrc);
      REQUIRE(dvlrc == -delrc);
   }
   mutant::vlambda = saved;
}

// What the long range correction adds to the energy, its lambda derivatives,
// the virial and its lambda derivative, as the difference of the fixture's key
// with and without its correction keyword, against Fortran evcorr1 carried to
// the main lambda. The key keeps only the term that owns the correction; its
// cubic van der Waals map puts both map derivatives into the chain rule.
void runLrcChainFixture(const Fixture& fx, bool dsp)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);
   TestFile fxyz(files + fx.xyz, xyzname);
   TestFile fprm(files + fx.prm);

   std::ifstream fin(files + "testlmda/" + fx.name + ".key");
   REQUIRE(fin.good());
   const char* corr = dsp ? "disp-correction" : "vdw-correction";
   std::string with, without, line;
   bool found = false;
   while (std::getline(fin, line)) {
      with += line + "\n";
      if (line.rfind(corr, 0) == 0)
         found = true;
      else
         without += line + "\n";
   }
   REQUIRE(found);
   const std::string only = dsp ? "dispersionterm only\n" : "vdwterm only\n";
   with += only;
   without += only;

   struct Terms
   {
      double e, dedl, d2edl2, vir[9], dvirdl[9];
   };
   double want[5] = {};
   auto run = [&](const std::string& content, bool expect) {
      const std::string keyname = std::string(fx.name) + "_lrc.key";
      TestFile fkey("", keyname, content);
      const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
      int argc = 4;
      TestSession session(argc, argv);
      FdTestOptions opts;
      opts.analyt = true;
      opts.numer = false;
      rc_flag = testlmdaFlags(opts);
      session.init();

      energy(calc::v1);
      // energy() maps the main lambda onto the sub-lambdas.
      REQUIRE(use_d2lmda);
      REQUIRE(dvldlmda != 0);
      REQUIRE(d2vldlmda2 != 0);
      Terms t;
      t.e = esum;
      t.dedl = dedl;
      t.d2edl2 = d2edl2;
      for (int k = 0; k < 9; ++k) {
         t.vir[k] = vir[k];
         t.dvirdl[k] = dvirdl[k];
      }

      if (expect) {
         // Fortran at the vdw lambda tinker9 holds.
         const double saved = mutant::vlambda;
         mutant::vlambda = vlam;
         double elrc, vlrc, delrc, dvlrc, d2elrc;
         if (dsp)
            tinker_f_evcorr1({const_cast<char*>("DISP  "), 6}, &elrc, &vlrc, &delrc, &dvlrc, &d2elrc);
         else
            tinker_f_evcorr1({const_cast<char*>("VDW   "), 6}, &elrc, &vlrc, &delrc, &dvlrc, &d2elrc);
         mutant::vlambda = saved;
         want[0] = elrc;
         want[1] = delrc * dvldlmda;
         want[2] = d2elrc * dvldlmda * dvldlmda + delrc * d2vldlmda2;
         want[3] = vlrc;
         want[4] = dvlrc * dvldlmda;
         REQUIRE(want[1] != 0);
         REQUIRE(want[2] != 0);
      }
      return t;
   };
   const Terms a = run(with, true);
   const Terms b = run(without, false);

   // The box volume is a real, which leaves the correction float round-off
   // apart from Fortran in mixed precision.
   const double rel = testGetEps(1.0e-5, 1.0e-10);
   auto near = [&](double x, double y) { return std::fabs(x - y) <= rel * std::fabs(y) + 1.0e-12; };
   CAPTURE(fx.name, a.e - b.e, want[0], a.dedl - b.dedl, want[1], a.d2edl2 - b.d2edl2, want[2]);
   REQUIRE(near(a.e - b.e, want[0]));
   REQUIRE(near(a.dedl - b.dedl, want[1]));
   REQUIRE(near(a.d2edl2 - b.d2edl2, want[2]));
   for (int k = 0; k < 9; ++k) {
      const bool diag = (k == 0 or k == 4 or k == 8);
      CAPTURE(k, a.vir[k] - b.vir[k], a.dvirdl[k] - b.dvirdl[k], want[3], want[4]);
      REQUIRE(near(a.vir[k] - b.vir[k], diag ? want[3] : 0));
      REQUIRE(near(a.dvirdl[k] - b.dvirdl[k], diag ? want[4] : 0));
   }
}
}

TEST_CASE("TESTLMDA-01_water_adt_l05", "[ff][testlmda]") { runFixture(fx("01_water_adt_l05")); }
TEST_CASE("TESTLMDA-02_water_ast_l05", "[ff][testlmda]") { runFixture(fx("02_water_ast_l05")); }
TEST_CASE("TESTLMDA-03_water_adt_l06exp", "[ff][testlmda]") { runFixture(fx("03_water_adt_l06exp")); }
TEST_CASE("TESTLMDA-04_water_ast_l06exp", "[ff][testlmda]") { runFixture(fx("04_water_ast_l06exp")); }
TEST_CASE("TESTLMDA-05_water_ast_nodl_l05", "[ff][testlmda]") { runFixture(fx("05_water_ast_nodl_l05")); }
TEST_CASE("TESTLMDA-06_water_ast_vonly_l05", "[ff][testlmda]") { runFixture(fx("06_water_ast_vonly_l05")); }
TEST_CASE("TESTLMDA-07_water_ast_vcorr_l05", "[ff][testlmda]") { runFixture(fx("07_water_ast_vcorr_l05")); }
TEST_CASE("TESTLMDA-08_water_ast_vcorr_annih_l05", "[ff][testlmda]") { runFixture(fx("08_water_ast_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-09_water_ast_vcorr_l06exp", "[ff][testlmda]") { runFixture(fx("09_water_ast_vcorr_l06exp")); }
TEST_CASE("TESTLMDA-10_water_rels_vdwm_vcorr_annih_l05", "[ff][testlmda][rdt]") { runFixture(fx("10_water_rels_vdwm_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-11_water_rels_vdwm_vcorr_l050", "[ff][testlmda][rdt]") { runFixture(fx("11_water_rels_vdwm_vcorr_l050")); }
TEST_CASE("TESTLMDA-12_water_rels_lig1_l085", "[ff][testlmda][rdt]") { runFixture(fx("12_water_rels_lig1_l085")); }
TEST_CASE("TESTLMDA-13_water_rels_lig2_l015", "[ff][testlmda][rdt]") { runFixture(fx("13_water_rels_lig2_l015")); }
TEST_CASE("TESTLMDA-14_water_rels_lig1_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("14_water_rels_lig1_ne_l085")); }
TEST_CASE("TESTLMDA-15_water_rels_lig1_nlist_exf_l085", "[ff][testlmda][rdt]") { runFixture(fx("15_water_rels_lig1_nlist_exf_l085")); }
TEST_CASE("TESTLMDA-16_water_rels_lig1_st_l085", "[ff][testlmda][rdt]") { runFixture(fx("16_water_rels_lig1_st_l085")); }
TEST_CASE("TESTLMDA-17_water_rels_lig2_st_l015", "[ff][testlmda][rdt]") { runFixture(fx("17_water_rels_lig2_st_l015")); }
TEST_CASE("TESTLMDA-18_water_rels_lig1_st_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("18_water_rels_lig1_st_ne_l085")); }
TEST_CASE("TESTLMDA-19_water_rels_lig1_st_polonly_l085", "[ff][testlmda][rdt]") { runFixture(fx("19_water_rels_lig1_st_polonly_l085")); }
TEST_CASE("TESTLMDA-20_water_rels_lig1_dt_polonly_l078", "[ff][testlmda][rdt]") { runFixture(fx("20_water_rels_lig1_dt_polonly_l078")); }
TEST_CASE("TESTLMDA-21_water_rels_lig1_st_l088", "[ff][testlmda][rdt]") { runFixture(fx("21_water_rels_lig1_st_l088")); }
TEST_CASE("TESTLMDA-22_water_rels_lig1_dt_l088", "[ff][testlmda][rdt]") { runFixture(fx("22_water_rels_lig1_dt_l088")); }
TEST_CASE("TESTLMDA-23_water_rels_3stg_l085", "[ff][testlmda][rdt]") { runFixture(fx("23_water_rels_3stg_l085")); }
TEST_CASE("TESTLMDA-24_ionwat_ewald_l05", "[ff][testlmda]") { runFixture(fx("24_ionwat_ewald_l05")); }
TEST_CASE("TESTLMDA-25_ionwat_ewald_nlist_l05", "[ff][testlmda]") { runFixture(fx("25_ionwat_ewald_nlist_l05")); }
TEST_CASE("TESTLMDA-26_ionwat_pol_ewald_l05", "[ff][testlmda]") { runFixture(fx("26_ionwat_pol_ewald_l05")); }
TEST_CASE("TESTLMDA-27_water_rels_2stg_l075", "[ff][testlmda][rdt]") { runFixture(fx("27_water_rels_2stg_l075")); }
TEST_CASE("TESTLMDA-28_hipwat_flux_l04", "[ff][testlmda][cflux]") { runFixture(fx("28_hipwat_flux_l04")); }
TEST_CASE("TESTLMDA-29_hipwat_flux_ewald_l04", "[ff][testlmda][cflux]") { runFixture(fx("29_hipwat_flux_ewald_l04")); }
TEST_CASE("TESTLMDA-30_hipwat_flux_ewald_nlist_l04", "[ff][testlmda][cflux]") { runFixture(fx("30_hipwat_flux_ewald_nlist_l04")); }
TEST_CASE("TESTLMDA-31_hipwat_flux_ewald_lin_l00", "[ff][testlmda][cflux]") { runFixture(fx("31_hipwat_flux_ewald_lin_l00")); }
TEST_CASE("TESTLMDA-32_aptetra_flux_l04", "[ff][testlmda][cflux]") { runFixture(fx("32_aptetra_flux_l04")); }
TEST_CASE("TESTLMDA-33_aptetra_flux_ewald_l04", "[ff][testlmda][cflux]") { runFixture(fx("33_aptetra_flux_ewald_l04")); }
TEST_CASE("TESTLMDA-34_aptetra_flux_ewald_nlist_l04", "[ff][testlmda][cflux]") { runFixture(fx("34_aptetra_flux_ewald_nlist_l04")); }
TEST_CASE("TESTLMDA-35_aptetra_flux_ewald_lin_l00", "[ff][testlmda][cflux]") { runFixture(fx("35_aptetra_flux_ewald_lin_l00")); }
TEST_CASE("TESTLMDA-36_apgas_flux_l04", "[ff][testlmda][cflux]") { runFixture(fx("36_apgas_flux_l04")); }
TEST_CASE("TESTLMDA-37_apgas_cross_l04", "[ff][testlmda][cflux]") { runFixture(fx("37_apgas_cross_l04")); }
TEST_CASE("TESTLMDA-38_apgas_cross_lin_l00", "[ff][testlmda][cflux]") { runFixture(fx("38_apgas_cross_lin_l00")); }
TEST_CASE("TESTLMDA-39_apgas_cross_ewald_l04", "[ff][testlmda][cflux]") { runFixture(fx("39_apgas_cross_ewald_l04")); }
TEST_CASE("TESTLMDA-40_hipmix_ewald_l04", "[ff][testlmda][cflux]") { runFixture(fx("40_hipmix_ewald_l04")); }
TEST_CASE("TESTLMDA-58_hipwat_flux_mlist_l04", "[ff][testlmda][cflux]") { runFixture(fx("58_hipwat_flux_mlist_l04")); }
TEST_CASE("TESTLMDA-41_hipwat_pol_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("41_hipwat_pol_st_l04")); }
TEST_CASE("TESTLMDA-42_hipwat_pol_ewald_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("42_hipwat_pol_ewald_st_l04")); }
TEST_CASE("TESTLMDA-43_hipwat_pol_ewald_nlist_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("43_hipwat_pol_ewald_nlist_st_l04")); }
TEST_CASE("TESTLMDA-44_hipwat_pol_ewald_lin_st_l00", "[ff][testlmda][cflux][astpol]") { runFixture(fx("44_hipwat_pol_ewald_lin_st_l00")); }
TEST_CASE("TESTLMDA-46_aptetra_pol_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("46_aptetra_pol_st_l04")); }
TEST_CASE("TESTLMDA-47_aptetra_pol_ewald_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("47_aptetra_pol_ewald_st_l04")); }
TEST_CASE("TESTLMDA-48_aptetra_pol_ewald_nlist_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("48_aptetra_pol_ewald_nlist_st_l04")); }
TEST_CASE("TESTLMDA-49_aptetra_pol_ewald_lin_st_l00", "[ff][testlmda][cflux][astpol]") { runFixture(fx("49_aptetra_pol_ewald_lin_st_l00")); }
TEST_CASE("TESTLMDA-51_apgas_pol_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("51_apgas_pol_st_l04")); }
TEST_CASE("TESTLMDA-53_hipmix_pol_ewald_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("53_hipmix_pol_ewald_st_l04")); }
TEST_CASE("TESTLMDA-57_hipwat_pol_split_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("57_hipwat_pol_split_st_l04")); }
TEST_CASE("TESTLMDA-45_hipwat_pol_ewald_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("45_hipwat_pol_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-50_aptetra_pol_ewald_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("50_aptetra_pol_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-52_apgas_pol_ewald_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("52_apgas_pol_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-54_hipmix_pol_ewald_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("54_hipmix_pol_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-56_hipwat_pol_split_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("56_hipwat_pol_split_dt_l04")); }
TEST_CASE("TESTLMDA-59_aptetra_pol_mlist_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("59_aptetra_pol_mlist_dt_l04")); }
TEST_CASE("TESTLMDA-60_aptetra_rel_lig1_l085", "[ff][testlmda][cflux][rdt]") { runFixture(fx("60_aptetra_rel_lig1_l085")); }
TEST_CASE("TESTLMDA-61_aptetra_rel_lig2_l015", "[ff][testlmda][cflux][rdt]") { runFixture(fx("61_aptetra_rel_lig2_l015")); }
TEST_CASE("TESTLMDA-62_aptetra_rel_cross_l085", "[ff][testlmda][cflux][rdt]") { runFixture(fx("62_aptetra_rel_cross_l085")); }
TEST_CASE("TESTLMDA-63_hipwat_rel_lig1_ewald_l085", "[ff][testlmda][cflux][rdt]") { runFixture(fx("63_hipwat_rel_lig1_ewald_l085")); }
// HIPPO water polarization driven alone (test_testlmda.f cases 64, 65).
TEST_CASE("TESTLMDA-64_hipwat_pol_only_st_l04", "[ff][testlmda][cflux][astpol]") { runFixture(fx("64_hipwat_pol_only_st_l04")); }
TEST_CASE("TESTLMDA-65_hipwat_pol_only_dt_l04", "[ff][testlmda][cflux][adtpol]") { runFixture(fx("65_hipwat_pol_only_dt_l04")); }
// Exchange polarization of an Na+ water cluster and an ion box, the ion or a
// water mutated, in dual and single topology (test_testlmda.f cases 66 to 72).
TEST_CASE("TESTLMDA-66_expnaclus_na_dt_l04", "[ff][testlmda][expol][adtpol]") { runFixture(fx("66_expnaclus_na_dt_l04")); }
TEST_CASE("TESTLMDA-67_expnaclus_wat_dt_l04", "[ff][testlmda][expol][adtpol]") { runFixture(fx("67_expnaclus_wat_dt_l04")); }
TEST_CASE("TESTLMDA-68_expbox_ewald_dt_l04", "[ff][testlmda][expol][adtpol]") { runFixture(fx("68_expbox_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-69_expnaclus_wat_st_l04", "[ff][testlmda][expol][astpol]") { runFixture(fx("69_expnaclus_wat_st_l04")); }
TEST_CASE("TESTLMDA-70_expbox_ewald_st_l04", "[ff][testlmda][expol][astpol]") { runFixture(fx("70_expbox_ewald_st_l04")); }
TEST_CASE("TESTLMDA-71_expnaclus_na_lin_st_l00", "[ff][testlmda][expol][astpol]") { runFixture(fx("71_expnaclus_na_lin_st_l00")); }
TEST_CASE("TESTLMDA-72_expbox_ewald_nlist_st_l04", "[ff][testlmda][expol][astpol]") { runFixture(fx("72_expbox_ewald_nlist_st_l04")); }
// The exchange polarization overlap that single topology gathers with each
// solve starts from zero every time.
TEST_CASE("TESTLMDA-repeat-70_expbox_ewald_st_l04", "[ff][testlmda][expol][astpol]") { runRepeatedFixture(fx("70_expbox_ewald_st_l04")); }
// Charge transfer follows the polarization lambda (test_testlmda.f cases 73 to
// 90). Case 81 needs replica images, which tinker9 does not have, and 82 weights
// atom groups, which tinker9 refuses for every term.
TEST_CASE("TESTLMDA-73_hipwat_ct_only_st_l04", "[ff][testlmda][ct]") { runFixture(fx("73_hipwat_ct_only_st_l04")); }
TEST_CASE("TESTLMDA-74_hipwat_ct_only_dt_l04", "[ff][testlmda][ct]") { runFixture(fx("74_hipwat_ct_only_dt_l04")); }
TEST_CASE("TESTLMDA-75_hipwat_ct_only_dtexp_l04", "[ff][testlmda][ct]") { runFixture(fx("75_hipwat_ct_only_dtexp_l04")); }
TEST_CASE("TESTLMDA-76_hipwat_ct_mlist_dt_l04", "[ff][testlmda][ct]") { runFixture(fx("76_hipwat_ct_mlist_dt_l04")); }
TEST_CASE("TESTLMDA-77_hipwat_ct_lin_dt_l00", "[ff][testlmda][ct]") { runFixture(fx("77_hipwat_ct_lin_dt_l00")); }
TEST_CASE("TESTLMDA-78_hipwat_pol_ct_split_st_l04", "[ff][testlmda][ct]") { runFixture(fx("78_hipwat_pol_ct_split_st_l04")); }
TEST_CASE("TESTLMDA-79_hipwat_pol_ct_ewald_dt_l04", "[ff][testlmda][ct]") { runFixture(fx("79_hipwat_pol_ct_ewald_dt_l04")); }
TEST_CASE("TESTLMDA-80_aptetra_ct_st_l04", "[ff][testlmda][ct]") { runFixture(fx("80_aptetra_ct_st_l04")); }
TEST_CASE("TESTLMDA-82_hipwat_ct_group_ost_l04", "[ff][testlmda][ct]") { runRefusedFixture(fx("82_hipwat_ct_group_ost_l04")); }
TEST_CASE("TESTLMDA-83_hipwat_ct_off_st_l04", "[ff][testlmda][ct]") { runFixture(fx("83_hipwat_ct_off_st_l04")); }
TEST_CASE("TESTLMDA-84_aptetra_ct_rel_lig1_st_l085", "[ff][testlmda][ct][rdt]") { runFixture(fx("84_aptetra_ct_rel_lig1_st_l085")); }
TEST_CASE("TESTLMDA-85_aptetra_ct_rel_lig2_st_l015", "[ff][testlmda][ct][rdt]") { runFixture(fx("85_aptetra_ct_rel_lig2_st_l015")); }
TEST_CASE("TESTLMDA-86_aptetra_ct_rel_lig1_dt_l080", "[ff][testlmda][ct][rdt]") { runFixture(fx("86_aptetra_ct_rel_lig1_dt_l080")); }
TEST_CASE("TESTLMDA-87_aptetra_ct_rel_lig2_dt_l020", "[ff][testlmda][ct][rdt]") { runFixture(fx("87_aptetra_ct_rel_lig2_dt_l020")); }
TEST_CASE("TESTLMDA-88_aptetra_ct_rel_vdwm_l050", "[ff][testlmda][ct][rdt]") { runFixture(fx("88_aptetra_ct_rel_vdwm_l050")); }
TEST_CASE("TESTLMDA-89_aptetra_pol_ct_rel_lig1_ewald_l080", "[ff][testlmda][ct][rdt]") { runFixture(fx("89_aptetra_pol_ct_rel_lig1_ewald_l080")); }
TEST_CASE("TESTLMDA-90_hipwat_pol_ct_rel_lig1_ewald_l080", "[ff][testlmda][ct][rdt]") { runFixture(fx("90_hipwat_pol_ct_rel_lig1_ewald_l080")); }
// HIPPO Pauli repulsion follows the van der Waals lambda (test_testlmda.f cases 91 to 100).
TEST_CASE("TESTLMDA-91_hipmix_rep_l04", "[ff][testlmda][repel]") { runFixture(fx("91_hipmix_rep_l04")); }
TEST_CASE("TESTLMDA-92_hipmix_rep_nlist_l04", "[ff][testlmda][repel]") { runFixture(fx("92_hipmix_rep_nlist_l04")); }
TEST_CASE("TESTLMDA-93_hipmix_rep_annih_l04", "[ff][testlmda][repel]") { runFixture(fx("93_hipmix_rep_annih_l04")); }
TEST_CASE("TESTLMDA-94_hipmix_rep_taper_l04", "[ff][testlmda][repel]") { runFixture(fx("94_hipmix_rep_taper_l04")); }
TEST_CASE("TESTLMDA-95_hipmix_rep_sc_m0p2_l04", "[ff][testlmda][repel]") { runFixture(fx("95_hipmix_rep_sc_m0p2_l04")); }
TEST_CASE("TESTLMDA-96_hipmix_rep_sc_real_l04", "[ff][testlmda][repel]") { runFixture(fx("96_hipmix_rep_sc_real_l04")); }
TEST_CASE("TESTLMDA-97_hipmix_rep_l00", "[ff][testlmda][repel]") { runFixture(fx("97_hipmix_rep_l00")); }
TEST_CASE("TESTLMDA-98_hipmix_rep_l10", "[ff][testlmda][repel]") { runFixture(fx("98_hipmix_rep_l10")); }
TEST_CASE("TESTLMDA-99_hipmix_rep_sc_m0p2_l10", "[ff][testlmda][repel]") { runFixture(fx("99_hipmix_rep_sc_m0p2_l10")); }
TEST_CASE("TESTLMDA-100_hipmix_rep_d1_n12_l04", "[ff][testlmda][repel]") { runFixture(fx("100_hipmix_rep_d1_n12_l04")); }
// A cubic van der Waals lambda map, so the in-kernel chain rule factors dv/dL
// and d2v/dL2 are neither one nor zero (test_testlmda.f case 117); the map is
// steep at lambda 0.8, so the step is halved to keep numerical d2E/dL2 close.
TEST_CASE("TESTLMDA-117_hipmix_rep_exp3_l08", "[ff][testlmda][repel]") { runFixture(fx("117_hipmix_rep_exp3_l08")); }
TEST_CASE("TESTLMDA-repeat-94_hipmix_rep_taper_l04", "[ff][testlmda][repel]") { runRepeatedFixture(fx("94_hipmix_rep_taper_l04")); }
// HIPPO damped dispersion follows the van der Waals lambda (test_testlmda.f
// cases 101 to 112). Case 111 needs replica images, which tinker9 does not
// have, and 112 weights atom groups, which tinker9 refuses.
TEST_CASE("TESTLMDA-101_hipmix_dsp_l04", "[ff][testlmda][disp]") { runFixture(fx("101_hipmix_dsp_l04")); }
TEST_CASE("TESTLMDA-102_hipmix_dsp_nlist_l04", "[ff][testlmda][disp]") { runFixture(fx("102_hipmix_dsp_nlist_l04")); }
TEST_CASE("TESTLMDA-103_hipmix_dsp_annih_l04", "[ff][testlmda][disp]") { runFixture(fx("103_hipmix_dsp_annih_l04")); }
TEST_CASE("TESTLMDA-104_hipmix_dsp_taper_l04", "[ff][testlmda][disp]") { runFixture(fx("104_hipmix_dsp_taper_l04")); }
TEST_CASE("TESTLMDA-105_hipmix_dsp_ewald_l04", "[ff][testlmda][disp]") { runFixture(fx("105_hipmix_dsp_ewald_l04")); }
TEST_CASE("TESTLMDA-106_hipmix_dsp_ewald_nlist_annih_l04", "[ff][testlmda][disp]") { runFixture(fx("106_hipmix_dsp_ewald_nlist_annih_l04")); }
TEST_CASE("TESTLMDA-107_hipmix_dsp_sc_real_l04", "[ff][testlmda][disp]") { runFixture(fx("107_hipmix_dsp_sc_real_l04")); }
TEST_CASE("TESTLMDA-108_hipmix_dsp_ewald_n2_l00", "[ff][testlmda][disp]") { runFixture(fx("108_hipmix_dsp_ewald_n2_l00")); }
TEST_CASE("TESTLMDA-109_hipmix_dsp_ewald_l10", "[ff][testlmda][disp]") { runFixture(fx("109_hipmix_dsp_ewald_l10")); }
TEST_CASE("TESTLMDA-110_hipmix_dsp_d1_n1_l04", "[ff][testlmda][disp]") { runFixture(fx("110_hipmix_dsp_d1_n1_l04")); }
// The reciprocal space lambda derivative of the force reuses the PME grid.
TEST_CASE("TESTLMDA-repeat-105_hipmix_dsp_ewald_l04", "[ff][testlmda][disp]") { runRepeatedFixture(fx("105_hipmix_dsp_ewald_l04")); }
TEST_CASE("TESTLMDA-112_hipmix_dsp_ewald_group_l04", "[ff][testlmda][disp]") { runRefusedFixture(fx("112_hipmix_dsp_ewald_group_l04")); }
// The long range dispersion correction follows the lambda power of the soft
// core, alone, under annihilation, with an exponent of one, and with an
// exponent of two at lambda 0 (test_testlmda.f cases 113 to 116).
TEST_CASE("TESTLMDA-113_hipmix_dsp_corr_l04", "[ff][testlmda][disp]") { runFixture(fx("113_hipmix_dsp_corr_l04")); }
TEST_CASE("TESTLMDA-114_hipmix_dsp_corr_annih_l04", "[ff][testlmda][disp]") { runFixture(fx("114_hipmix_dsp_corr_annih_l04")); }
TEST_CASE("TESTLMDA-115_hipmix_dsp_corr_n1_l04", "[ff][testlmda][disp]") { runFixture(fx("115_hipmix_dsp_corr_n1_l04")); }
TEST_CASE("TESTLMDA-116_hipmix_dsp_corr_n2_l00", "[ff][testlmda][disp]") { runFixture(fx("116_hipmix_dsp_corr_n2_l00")); }
// Ewald dispersion through a cubic van der Waals lambda map, so the soft core
// powers of the real space, reciprocal space and self terms all carry both map
// derivatives (test_testlmda.f case 120); the step is halved as for 117.
// The class sums of the long range correction give Fortran's correction at any
// lambda: buffered 14-7 with ligand 1 decoupled or annihilated and with both
// ligands of a relative leg, and dispersion decoupled or annihilated.
TEST_CASE("TESTLMDA-lrc-evcorr1", "[ff][testlmda]")
{
   runLrcFixture(fx("07_water_ast_vcorr_l05"), false);
   runLrcFixture(fx("08_water_ast_vcorr_annih_l05"), false);
   runLrcFixture(fx("10_water_rels_vdwm_vcorr_annih_l05"), false);
   runLrcFixture(fx("11_water_rels_vdwm_vcorr_l050"), false);
   runLrcFixture(fx("113_hipmix_dsp_corr_l04"), true);
   runLrcFixture(fx("114_hipmix_dsp_corr_annih_l04"), true);
}
// The long range correction's own lambda derivatives and virial through a cubic
// van der Waals map, buffered 14-7 and dispersion at lambda 0.6.
TEST_CASE("TESTLMDA-lrc-chain", "[ff][testlmda]")
{
   runLrcChainFixture(fx("09_water_ast_vcorr_l06exp"), false);
   runLrcChainFixture(fx("121_hipmix_dsp_corr_exp3_l06"), true);
}
TEST_CASE("TESTLMDA-120_hipmix_dsp_ewald_exp3_l08", "[ff][testlmda][disp]") { runFixture(fx("120_hipmix_dsp_ewald_exp3_l08")); }
// The long range dispersion correction through the cubic map (test_testlmda.f
// case 121), whose curvature puts d2vl into the chain rule.
TEST_CASE("TESTLMDA-121_hipmix_dsp_corr_exp3_l06", "[ff][testlmda][disp]") { runFixture(fx("121_hipmix_dsp_corr_exp3_l06")); }
// The other van der Waals map shapes, a quintic taper and an inverse power,
// through repulsion and dispersion (test_testlmda.f cases 122 to 125).
TEST_CASE("TESTLMDA-122_hipmix_rep_qnt_l045", "[ff][testlmda][repel]") { runFixture(fx("122_hipmix_rep_qnt_l045")); }
TEST_CASE("TESTLMDA-123_hipmix_rep_inv_l03", "[ff][testlmda][repel]") { runFixture(fx("123_hipmix_rep_inv_l03")); }
TEST_CASE("TESTLMDA-124_hipmix_dsp_corr_qnt_l06", "[ff][testlmda][disp]") { runFixture(fx("124_hipmix_dsp_corr_qnt_l06")); }
TEST_CASE("TESTLMDA-125_hipmix_dsp_ewald_inv_l04", "[ff][testlmda][disp]") { runFixture(fx("125_hipmix_dsp_ewald_inv_l04")); }
// The buffered 14-7 soft core with a softening of alpha*lambda^m*(1-lambda)^p,
// m 1.5 and p 3, then a non-integer m of 0.5 and p of one (test_testlmda.f
// cases 118 and 119).
TEST_CASE("TESTLMDA-118_water_vsoft_m15_p3_l05", "[ff][testlmda]") { runFixture(fx("118_water_vsoft_m15_p3_l05")); }
TEST_CASE("TESTLMDA-119_water_vsoft_m05_p1_l03", "[ff][testlmda]") { runFixture(fx("119_water_vsoft_m05_p1_l03")); }

// Every fixture once more without analysis, as dynamics runs it.
TEST_CASE("TESTLMDA-DYN-01_water_adt_l05", "[ff][testlmda][dyn]") { runFixture(fx("01_water_adt_l05"), false); }
TEST_CASE("TESTLMDA-DYN-02_water_ast_l05", "[ff][testlmda][dyn]") { runFixture(fx("02_water_ast_l05"), false); }
TEST_CASE("TESTLMDA-DYN-03_water_adt_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("03_water_adt_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-04_water_ast_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("04_water_ast_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-05_water_ast_nodl_l05", "[ff][testlmda][dyn]") { runFixture(fx("05_water_ast_nodl_l05"), false); }
TEST_CASE("TESTLMDA-DYN-06_water_ast_vonly_l05", "[ff][testlmda][dyn]") { runFixture(fx("06_water_ast_vonly_l05"), false); }
TEST_CASE("TESTLMDA-DYN-07_water_ast_vcorr_l05", "[ff][testlmda][dyn]") { runFixture(fx("07_water_ast_vcorr_l05"), false); }
TEST_CASE("TESTLMDA-DYN-08_water_ast_vcorr_annih_l05", "[ff][testlmda][dyn]") { runFixture(fx("08_water_ast_vcorr_annih_l05"), false); }
TEST_CASE("TESTLMDA-DYN-09_water_ast_vcorr_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("09_water_ast_vcorr_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-10_water_rels_vdwm_vcorr_annih_l05", "[ff][testlmda][dyn][rdt]") { runFixture(fx("10_water_rels_vdwm_vcorr_annih_l05"), false); }
TEST_CASE("TESTLMDA-DYN-11_water_rels_vdwm_vcorr_l050", "[ff][testlmda][dyn][rdt]") { runFixture(fx("11_water_rels_vdwm_vcorr_l050"), false); }
TEST_CASE("TESTLMDA-DYN-12_water_rels_lig1_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("12_water_rels_lig1_l085"), false); }
TEST_CASE("TESTLMDA-DYN-13_water_rels_lig2_l015", "[ff][testlmda][dyn][rdt]") { runFixture(fx("13_water_rels_lig2_l015"), false); }
TEST_CASE("TESTLMDA-DYN-14_water_rels_lig1_ne_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("14_water_rels_lig1_ne_l085"), false); }
TEST_CASE("TESTLMDA-DYN-15_water_rels_lig1_nlist_exf_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("15_water_rels_lig1_nlist_exf_l085"), false); }
TEST_CASE("TESTLMDA-DYN-16_water_rels_lig1_st_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("16_water_rels_lig1_st_l085"), false); }
TEST_CASE("TESTLMDA-DYN-17_water_rels_lig2_st_l015", "[ff][testlmda][dyn][rdt]") { runFixture(fx("17_water_rels_lig2_st_l015"), false); }
TEST_CASE("TESTLMDA-DYN-18_water_rels_lig1_st_ne_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("18_water_rels_lig1_st_ne_l085"), false); }
TEST_CASE("TESTLMDA-DYN-19_water_rels_lig1_st_polonly_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("19_water_rels_lig1_st_polonly_l085"), false); }
TEST_CASE("TESTLMDA-DYN-20_water_rels_lig1_dt_polonly_l078", "[ff][testlmda][dyn][rdt]") { runFixture(fx("20_water_rels_lig1_dt_polonly_l078"), false); }
TEST_CASE("TESTLMDA-DYN-21_water_rels_lig1_st_l088", "[ff][testlmda][dyn][rdt]") { runFixture(fx("21_water_rels_lig1_st_l088"), false); }
TEST_CASE("TESTLMDA-DYN-22_water_rels_lig1_dt_l088", "[ff][testlmda][dyn][rdt]") { runFixture(fx("22_water_rels_lig1_dt_l088"), false); }
TEST_CASE("TESTLMDA-DYN-23_water_rels_3stg_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("23_water_rels_3stg_l085"), false); }
TEST_CASE("TESTLMDA-DYN-24_ionwat_ewald_l05", "[ff][testlmda][dyn]") { runFixture(fx("24_ionwat_ewald_l05"), false); }
TEST_CASE("TESTLMDA-DYN-25_ionwat_ewald_nlist_l05", "[ff][testlmda][dyn]") { runFixture(fx("25_ionwat_ewald_nlist_l05"), false); }
TEST_CASE("TESTLMDA-DYN-26_ionwat_pol_ewald_l05", "[ff][testlmda][dyn]") { runFixture(fx("26_ionwat_pol_ewald_l05"), false); }
TEST_CASE("TESTLMDA-DYN-27_water_rels_2stg_l075", "[ff][testlmda][dyn][rdt]") { runFixture(fx("27_water_rels_2stg_l075"), false); }
TEST_CASE("TESTLMDA-DYN-28_hipwat_flux_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("28_hipwat_flux_l04"), false); }
TEST_CASE("TESTLMDA-DYN-29_hipwat_flux_ewald_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("29_hipwat_flux_ewald_l04"), false); }
TEST_CASE("TESTLMDA-DYN-30_hipwat_flux_ewald_nlist_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("30_hipwat_flux_ewald_nlist_l04"), false); }
TEST_CASE("TESTLMDA-DYN-31_hipwat_flux_ewald_lin_l00", "[ff][testlmda][dyn][cflux]") { runFixture(fx("31_hipwat_flux_ewald_lin_l00"), false); }
TEST_CASE("TESTLMDA-DYN-32_aptetra_flux_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("32_aptetra_flux_l04"), false); }
TEST_CASE("TESTLMDA-DYN-33_aptetra_flux_ewald_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("33_aptetra_flux_ewald_l04"), false); }
TEST_CASE("TESTLMDA-DYN-34_aptetra_flux_ewald_nlist_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("34_aptetra_flux_ewald_nlist_l04"), false); }
TEST_CASE("TESTLMDA-DYN-35_aptetra_flux_ewald_lin_l00", "[ff][testlmda][dyn][cflux]") { runFixture(fx("35_aptetra_flux_ewald_lin_l00"), false); }
TEST_CASE("TESTLMDA-DYN-36_apgas_flux_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("36_apgas_flux_l04"), false); }
TEST_CASE("TESTLMDA-DYN-37_apgas_cross_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("37_apgas_cross_l04"), false); }
TEST_CASE("TESTLMDA-DYN-38_apgas_cross_lin_l00", "[ff][testlmda][dyn][cflux]") { runFixture(fx("38_apgas_cross_lin_l00"), false); }
TEST_CASE("TESTLMDA-DYN-39_apgas_cross_ewald_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("39_apgas_cross_ewald_l04"), false); }
TEST_CASE("TESTLMDA-DYN-40_hipmix_ewald_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("40_hipmix_ewald_l04"), false); }
TEST_CASE("TESTLMDA-DYN-58_hipwat_flux_mlist_l04", "[ff][testlmda][dyn][cflux]") { runFixture(fx("58_hipwat_flux_mlist_l04"), false); }
TEST_CASE("TESTLMDA-DYN-41_hipwat_pol_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("41_hipwat_pol_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-42_hipwat_pol_ewald_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("42_hipwat_pol_ewald_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-43_hipwat_pol_ewald_nlist_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("43_hipwat_pol_ewald_nlist_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-44_hipwat_pol_ewald_lin_st_l00", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("44_hipwat_pol_ewald_lin_st_l00"), false); }
TEST_CASE("TESTLMDA-DYN-46_aptetra_pol_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("46_aptetra_pol_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-47_aptetra_pol_ewald_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("47_aptetra_pol_ewald_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-48_aptetra_pol_ewald_nlist_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("48_aptetra_pol_ewald_nlist_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-49_aptetra_pol_ewald_lin_st_l00", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("49_aptetra_pol_ewald_lin_st_l00"), false); }
TEST_CASE("TESTLMDA-DYN-51_apgas_pol_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("51_apgas_pol_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-53_hipmix_pol_ewald_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("53_hipmix_pol_ewald_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-57_hipwat_pol_split_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("57_hipwat_pol_split_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-64_hipwat_pol_only_st_l04", "[ff][testlmda][dyn][cflux][astpol]") { runFixture(fx("64_hipwat_pol_only_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-65_hipwat_pol_only_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("65_hipwat_pol_only_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-45_hipwat_pol_ewald_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("45_hipwat_pol_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-50_aptetra_pol_ewald_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("50_aptetra_pol_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-52_apgas_pol_ewald_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("52_apgas_pol_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-54_hipmix_pol_ewald_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("54_hipmix_pol_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-56_hipwat_pol_split_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("56_hipwat_pol_split_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-59_aptetra_pol_mlist_dt_l04", "[ff][testlmda][dyn][cflux][adtpol]") { runFixture(fx("59_aptetra_pol_mlist_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-60_aptetra_rel_lig1_l085", "[ff][testlmda][dyn][cflux][rdt]") { runFixture(fx("60_aptetra_rel_lig1_l085"), false); }
TEST_CASE("TESTLMDA-DYN-61_aptetra_rel_lig2_l015", "[ff][testlmda][dyn][cflux][rdt]") { runFixture(fx("61_aptetra_rel_lig2_l015"), false); }
TEST_CASE("TESTLMDA-DYN-62_aptetra_rel_cross_l085", "[ff][testlmda][dyn][cflux][rdt]") { runFixture(fx("62_aptetra_rel_cross_l085"), false); }
TEST_CASE("TESTLMDA-DYN-63_hipwat_rel_lig1_ewald_l085", "[ff][testlmda][dyn][cflux][rdt]") { runFixture(fx("63_hipwat_rel_lig1_ewald_l085"), false); }
TEST_CASE("TESTLMDA-DYN-66_expnaclus_na_dt_l04", "[ff][testlmda][dyn][expol][adtpol]") { runFixture(fx("66_expnaclus_na_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-67_expnaclus_wat_dt_l04", "[ff][testlmda][dyn][expol][adtpol]") { runFixture(fx("67_expnaclus_wat_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-68_expbox_ewald_dt_l04", "[ff][testlmda][dyn][expol][adtpol]") { runFixture(fx("68_expbox_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-69_expnaclus_wat_st_l04", "[ff][testlmda][dyn][expol][astpol]") { runFixture(fx("69_expnaclus_wat_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-70_expbox_ewald_st_l04", "[ff][testlmda][dyn][expol][astpol]") { runFixture(fx("70_expbox_ewald_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-71_expnaclus_na_lin_st_l00", "[ff][testlmda][dyn][expol][astpol]") { runFixture(fx("71_expnaclus_na_lin_st_l00"), false); }
TEST_CASE("TESTLMDA-DYN-72_expbox_ewald_nlist_st_l04", "[ff][testlmda][dyn][expol][astpol]") { runFixture(fx("72_expbox_ewald_nlist_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-73_hipwat_ct_only_st_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("73_hipwat_ct_only_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-74_hipwat_ct_only_dt_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("74_hipwat_ct_only_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-75_hipwat_ct_only_dtexp_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("75_hipwat_ct_only_dtexp_l04"), false); }
TEST_CASE("TESTLMDA-DYN-76_hipwat_ct_mlist_dt_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("76_hipwat_ct_mlist_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-77_hipwat_ct_lin_dt_l00", "[ff][testlmda][dyn][ct]") { runFixture(fx("77_hipwat_ct_lin_dt_l00"), false); }
TEST_CASE("TESTLMDA-DYN-78_hipwat_pol_ct_split_st_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("78_hipwat_pol_ct_split_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-79_hipwat_pol_ct_ewald_dt_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("79_hipwat_pol_ct_ewald_dt_l04"), false); }
TEST_CASE("TESTLMDA-DYN-80_aptetra_ct_st_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("80_aptetra_ct_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-83_hipwat_ct_off_st_l04", "[ff][testlmda][dyn][ct]") { runFixture(fx("83_hipwat_ct_off_st_l04"), false); }
TEST_CASE("TESTLMDA-DYN-84_aptetra_ct_rel_lig1_st_l085", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("84_aptetra_ct_rel_lig1_st_l085"), false); }
TEST_CASE("TESTLMDA-DYN-85_aptetra_ct_rel_lig2_st_l015", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("85_aptetra_ct_rel_lig2_st_l015"), false); }
TEST_CASE("TESTLMDA-DYN-86_aptetra_ct_rel_lig1_dt_l080", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("86_aptetra_ct_rel_lig1_dt_l080"), false); }
TEST_CASE("TESTLMDA-DYN-87_aptetra_ct_rel_lig2_dt_l020", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("87_aptetra_ct_rel_lig2_dt_l020"), false); }
TEST_CASE("TESTLMDA-DYN-88_aptetra_ct_rel_vdwm_l050", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("88_aptetra_ct_rel_vdwm_l050"), false); }
TEST_CASE("TESTLMDA-DYN-89_aptetra_pol_ct_rel_lig1_ewald_l080", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("89_aptetra_pol_ct_rel_lig1_ewald_l080"), false); }
TEST_CASE("TESTLMDA-DYN-90_hipwat_pol_ct_rel_lig1_ewald_l080", "[ff][testlmda][dyn][ct][rdt]") { runFixture(fx("90_hipwat_pol_ct_rel_lig1_ewald_l080"), false); }
TEST_CASE("TESTLMDA-DYN-91_hipmix_rep_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("91_hipmix_rep_l04"), false); }
TEST_CASE("TESTLMDA-DYN-92_hipmix_rep_nlist_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("92_hipmix_rep_nlist_l04"), false); }
TEST_CASE("TESTLMDA-DYN-93_hipmix_rep_annih_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("93_hipmix_rep_annih_l04"), false); }
TEST_CASE("TESTLMDA-DYN-94_hipmix_rep_taper_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("94_hipmix_rep_taper_l04"), false); }
TEST_CASE("TESTLMDA-DYN-95_hipmix_rep_sc_m0p2_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("95_hipmix_rep_sc_m0p2_l04"), false); }
TEST_CASE("TESTLMDA-DYN-96_hipmix_rep_sc_real_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("96_hipmix_rep_sc_real_l04"), false); }
TEST_CASE("TESTLMDA-DYN-97_hipmix_rep_l00", "[ff][testlmda][dyn][repel]") { runFixture(fx("97_hipmix_rep_l00"), false); }
TEST_CASE("TESTLMDA-DYN-98_hipmix_rep_l10", "[ff][testlmda][dyn][repel]") { runFixture(fx("98_hipmix_rep_l10"), false); }
TEST_CASE("TESTLMDA-DYN-99_hipmix_rep_sc_m0p2_l10", "[ff][testlmda][dyn][repel]") { runFixture(fx("99_hipmix_rep_sc_m0p2_l10"), false); }
TEST_CASE("TESTLMDA-DYN-100_hipmix_rep_d1_n12_l04", "[ff][testlmda][dyn][repel]") { runFixture(fx("100_hipmix_rep_d1_n12_l04"), false); }
TEST_CASE("TESTLMDA-DYN-117_hipmix_rep_exp3_l08", "[ff][testlmda][dyn][repel]") { runFixture(fx("117_hipmix_rep_exp3_l08"), false); }
TEST_CASE("TESTLMDA-DYN-101_hipmix_dsp_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("101_hipmix_dsp_l04"), false); }
TEST_CASE("TESTLMDA-DYN-102_hipmix_dsp_nlist_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("102_hipmix_dsp_nlist_l04"), false); }
TEST_CASE("TESTLMDA-DYN-103_hipmix_dsp_annih_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("103_hipmix_dsp_annih_l04"), false); }
TEST_CASE("TESTLMDA-DYN-104_hipmix_dsp_taper_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("104_hipmix_dsp_taper_l04"), false); }
TEST_CASE("TESTLMDA-DYN-105_hipmix_dsp_ewald_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("105_hipmix_dsp_ewald_l04"), false); }
TEST_CASE("TESTLMDA-DYN-106_hipmix_dsp_ewald_nlist_annih_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("106_hipmix_dsp_ewald_nlist_annih_l04"), false); }
TEST_CASE("TESTLMDA-DYN-107_hipmix_dsp_sc_real_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("107_hipmix_dsp_sc_real_l04"), false); }
TEST_CASE("TESTLMDA-DYN-108_hipmix_dsp_ewald_n2_l00", "[ff][testlmda][dyn][disp]") { runFixture(fx("108_hipmix_dsp_ewald_n2_l00"), false); }
TEST_CASE("TESTLMDA-DYN-109_hipmix_dsp_ewald_l10", "[ff][testlmda][dyn][disp]") { runFixture(fx("109_hipmix_dsp_ewald_l10"), false); }
TEST_CASE("TESTLMDA-DYN-110_hipmix_dsp_d1_n1_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("110_hipmix_dsp_d1_n1_l04"), false); }
TEST_CASE("TESTLMDA-DYN-113_hipmix_dsp_corr_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("113_hipmix_dsp_corr_l04"), false); }
TEST_CASE("TESTLMDA-DYN-116_hipmix_dsp_corr_n2_l00", "[ff][testlmda][dyn][disp]") { runFixture(fx("116_hipmix_dsp_corr_n2_l00"), false); }
TEST_CASE("TESTLMDA-DYN-120_hipmix_dsp_ewald_exp3_l08", "[ff][testlmda][dyn][disp]") { runFixture(fx("120_hipmix_dsp_ewald_exp3_l08"), false); }
TEST_CASE("TESTLMDA-DYN-121_hipmix_dsp_corr_exp3_l06", "[ff][testlmda][dyn][disp]") { runFixture(fx("121_hipmix_dsp_corr_exp3_l06"), false); }
TEST_CASE("TESTLMDA-DYN-122_hipmix_rep_qnt_l045", "[ff][testlmda][dyn][repel]") { runFixture(fx("122_hipmix_rep_qnt_l045"), false); }
TEST_CASE("TESTLMDA-DYN-123_hipmix_rep_inv_l03", "[ff][testlmda][dyn][repel]") { runFixture(fx("123_hipmix_rep_inv_l03"), false); }
TEST_CASE("TESTLMDA-DYN-124_hipmix_dsp_corr_qnt_l06", "[ff][testlmda][dyn][disp]") { runFixture(fx("124_hipmix_dsp_corr_qnt_l06"), false); }
TEST_CASE("TESTLMDA-DYN-125_hipmix_dsp_ewald_inv_l04", "[ff][testlmda][dyn][disp]") { runFixture(fx("125_hipmix_dsp_ewald_inv_l04"), false); }
TEST_CASE("TESTLMDA-DYN-118_water_vsoft_m15_p3_l05", "[ff][testlmda][dyn]") { runFixture(fx("118_water_vsoft_m15_p3_l05"), false); }
TEST_CASE("TESTLMDA-DYN-119_water_vsoft_m05_p1_l03", "[ff][testlmda][dyn]") { runFixture(fx("119_water_vsoft_m05_p1_l03"), false); }
#endif
