#include "ff/modamoeba.h"
#include "seq/launch.h"

namespace tinker {
__global__
static void ulspredSaveP1_cu1(int n, real (*restrict ud)[3], real (*restrict up)[3], const real (*restrict uind)[3],
   const real (*restrict uinp)[3])
{
   if (uinp) {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         ud[i][0] = uind[i][0];
         ud[i][1] = uind[i][1];
         ud[i][2] = uind[i][2];
         up[i][0] = uinp[i][0];
         up[i][1] = uinp[i][1];
         up[i][2] = uinp[i][2];
      }
   } else {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         ud[i][0] = uind[i][0];
         ud[i][1] = uind[i][1];
         ud[i][2] = uind[i][2];
      }
   }
}

void ulspredSaveP1_cu(real (*ud)[3], real (*up)[3], const real (*uind)[3], const real (*uinp)[3])
{
   launch_k1s(g::s0, n, ulspredSaveP1_cu1, n, ud, up, uind, uinp);
}
}

namespace tinker {
__global__
static void ulspredSumASPC_cu(int n, int nualt,        //
   real (*restrict uind)[3], real (*restrict uinp)[3], //
   const real (*restrict udalt_00)[3], const real (*restrict udalt_01)[3], const real (*restrict udalt_02)[3],
   const real (*restrict udalt_03)[3], const real (*restrict udalt_04)[3], const real (*restrict udalt_05)[3],
   const real (*restrict udalt_06)[3], const real (*restrict udalt_07)[3], const real (*restrict udalt_08)[3],
   const real (*restrict udalt_09)[3], const real (*restrict udalt_10)[3], const real (*restrict udalt_11)[3],
   const real (*restrict udalt_12)[3], const real (*restrict udalt_13)[3], const real (*restrict udalt_14)[3],
   const real (*restrict udalt_15)[3], //
   const real (*restrict upalt_00)[3], const real (*restrict upalt_01)[3], const real (*restrict upalt_02)[3],
   const real (*restrict upalt_03)[3], const real (*restrict upalt_04)[3], const real (*restrict upalt_05)[3],
   const real (*restrict upalt_06)[3], const real (*restrict upalt_07)[3], const real (*restrict upalt_08)[3],
   const real (*restrict upalt_09)[3], const real (*restrict upalt_10)[3], const real (*restrict upalt_11)[3],
   const real (*restrict upalt_12)[3], const real (*restrict upalt_13)[3], const real (*restrict upalt_14)[3],
   const real (*restrict upalt_15)[3])
{
   constexpr double aspc[16] = {62. / 17., //
      -310. / 51.,                         //
      2170. / 323.,                        //
      -2329. / 400.,                       //
      1701. / 409.,                        //
      -806. / 323.,                        //
      1024. / 809.,                        //
      -479. / 883.,                        //
      257. / 1316.,                        //
      -434. / 7429.,                       //
      191. / 13375.,                       //
      -62. / 22287.,                       //
      3. / 7217.,                          //
      -3. / 67015.,                        //
      2. / 646323.,                        //
      -1. / 9694845.};

   double c00, c01, c02, c03, c04, c05, c06, c07;
   double c08, c09, c10, c11, c12, c13, c14, c15;
   c00 = aspc[(nualt - 1 + 16) % 16];
   c01 = aspc[(nualt - 2 + 16) % 16];
   c02 = aspc[(nualt - 3 + 16) % 16];
   c03 = aspc[(nualt - 4 + 16) % 16];
   c04 = aspc[(nualt - 5 + 16) % 16];
   c05 = aspc[(nualt - 6 + 16) % 16];
   c06 = aspc[(nualt - 7 + 16) % 16];
   c07 = aspc[(nualt - 8 + 16) % 16];
   c08 = aspc[(nualt - 9 + 16) % 16];
   c09 = aspc[(nualt - 10 + 16) % 16];
   c10 = aspc[(nualt - 11 + 16) % 16];
   c11 = aspc[(nualt - 12 + 16) % 16];
   c12 = aspc[(nualt - 13 + 16) % 16];
   c13 = aspc[(nualt - 14 + 16) % 16];
   c14 = aspc[(nualt - 15 + 16) % 16];
   c15 = aspc[(nualt - 16 + 16) % 16];

   if (uinp) {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         uind[i][0] = c00 * udalt_00[i][0] + c01 * udalt_01[i][0] + c02 * udalt_02[i][0] + c03 * udalt_03[i][0]
            + c04 * udalt_04[i][0] + c05 * udalt_05[i][0] + c06 * udalt_06[i][0] + c07 * udalt_07[i][0]
            + c08 * udalt_08[i][0] + c09 * udalt_09[i][0] + c10 * udalt_10[i][0] + c11 * udalt_11[i][0]
            + c12 * udalt_12[i][0] + c13 * udalt_13[i][0] + c14 * udalt_14[i][0] + c15 * udalt_15[i][0];
         uind[i][1] = c00 * udalt_00[i][1] + c01 * udalt_01[i][1] + c02 * udalt_02[i][1] + c03 * udalt_03[i][1]
            + c04 * udalt_04[i][1] + c05 * udalt_05[i][1] + c06 * udalt_06[i][1] + c07 * udalt_07[i][1]
            + c08 * udalt_08[i][1] + c09 * udalt_09[i][1] + c10 * udalt_10[i][1] + c11 * udalt_11[i][1]
            + c12 * udalt_12[i][1] + c13 * udalt_13[i][1] + c14 * udalt_14[i][1] + c15 * udalt_15[i][1];
         uind[i][2] = c00 * udalt_00[i][2] + c01 * udalt_01[i][2] + c02 * udalt_02[i][2] + c03 * udalt_03[i][2]
            + c04 * udalt_04[i][2] + c05 * udalt_05[i][2] + c06 * udalt_06[i][2] + c07 * udalt_07[i][2]
            + c08 * udalt_08[i][2] + c09 * udalt_09[i][2] + c10 * udalt_10[i][2] + c11 * udalt_11[i][2]
            + c12 * udalt_12[i][2] + c13 * udalt_13[i][2] + c14 * udalt_14[i][2] + c15 * udalt_15[i][2];
         uinp[i][0] = c00 * upalt_00[i][0] + c01 * upalt_01[i][0] + c02 * upalt_02[i][0] + c03 * upalt_03[i][0]
            + c04 * upalt_04[i][0] + c05 * upalt_05[i][0] + c06 * upalt_06[i][0] + c07 * upalt_07[i][0]
            + c08 * upalt_08[i][0] + c09 * upalt_09[i][0] + c10 * upalt_10[i][0] + c11 * upalt_11[i][0]
            + c12 * upalt_12[i][0] + c13 * upalt_13[i][0] + c14 * upalt_14[i][0] + c15 * upalt_15[i][0];
         uinp[i][1] = c00 * upalt_00[i][1] + c01 * upalt_01[i][1] + c02 * upalt_02[i][1] + c03 * upalt_03[i][1]
            + c04 * upalt_04[i][1] + c05 * upalt_05[i][1] + c06 * upalt_06[i][1] + c07 * upalt_07[i][1]
            + c08 * upalt_08[i][1] + c09 * upalt_09[i][1] + c10 * upalt_10[i][1] + c11 * upalt_11[i][1]
            + c12 * upalt_12[i][1] + c13 * upalt_13[i][1] + c14 * upalt_14[i][1] + c15 * upalt_15[i][1];
         uinp[i][2] = c00 * upalt_00[i][2] + c01 * upalt_01[i][2] + c02 * upalt_02[i][2] + c03 * upalt_03[i][2]
            + c04 * upalt_04[i][2] + c05 * upalt_05[i][2] + c06 * upalt_06[i][2] + c07 * upalt_07[i][2]
            + c08 * upalt_08[i][2] + c09 * upalt_09[i][2] + c10 * upalt_10[i][2] + c11 * upalt_11[i][2]
            + c12 * upalt_12[i][2] + c13 * upalt_13[i][2] + c14 * upalt_14[i][2] + c15 * upalt_15[i][2];
      }
   } else {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         uind[i][0] = c00 * udalt_00[i][0] + c01 * udalt_01[i][0] + c02 * udalt_02[i][0] + c03 * udalt_03[i][0]
            + c04 * udalt_04[i][0] + c05 * udalt_05[i][0] + c06 * udalt_06[i][0] + c07 * udalt_07[i][0]
            + c08 * udalt_08[i][0] + c09 * udalt_09[i][0] + c10 * udalt_10[i][0] + c11 * udalt_11[i][0]
            + c12 * udalt_12[i][0] + c13 * udalt_13[i][0] + c14 * udalt_14[i][0] + c15 * udalt_15[i][0];
         uind[i][1] = c00 * udalt_00[i][1] + c01 * udalt_01[i][1] + c02 * udalt_02[i][1] + c03 * udalt_03[i][1]
            + c04 * udalt_04[i][1] + c05 * udalt_05[i][1] + c06 * udalt_06[i][1] + c07 * udalt_07[i][1]
            + c08 * udalt_08[i][1] + c09 * udalt_09[i][1] + c10 * udalt_10[i][1] + c11 * udalt_11[i][1]
            + c12 * udalt_12[i][1] + c13 * udalt_13[i][1] + c14 * udalt_14[i][1] + c15 * udalt_15[i][1];
         uind[i][2] = c00 * udalt_00[i][2] + c01 * udalt_01[i][2] + c02 * udalt_02[i][2] + c03 * udalt_03[i][2]
            + c04 * udalt_04[i][2] + c05 * udalt_05[i][2] + c06 * udalt_06[i][2] + c07 * udalt_07[i][2]
            + c08 * udalt_08[i][2] + c09 * udalt_09[i][2] + c10 * udalt_10[i][2] + c11 * udalt_11[i][2]
            + c12 * udalt_12[i][2] + c13 * udalt_13[i][2] + c14 * udalt_14[i][2] + c15 * udalt_15[i][2];
      }
   }
}

__global__
static void ulspredSumGEAR_cu(int n, int nualt,        //
   real (*restrict uind)[3], real (*restrict uinp)[3], //
   const real (*restrict udalt_00)[3], const real (*restrict udalt_01)[3], const real (*restrict udalt_02)[3],
   const real (*restrict udalt_03)[3], const real (*restrict udalt_04)[3], const real (*restrict udalt_05)[3], //
   const real (*restrict upalt_00)[3], const real (*restrict upalt_01)[3], const real (*restrict upalt_02)[3],
   const real (*restrict upalt_03)[3], const real (*restrict upalt_04)[3], const real (*restrict upalt_05)[3])
{
   constexpr double gear[6] = {6., //
      -15.,                        //
      20.,                         //
      -15.,                        //
      6.,                          //
      -1.};

   double c00, c01, c02, c03, c04, c05;
   c00 = gear[(nualt - 1 + 6) % 6];
   c01 = gear[(nualt - 2 + 6) % 6];
   c02 = gear[(nualt - 3 + 6) % 6];
   c03 = gear[(nualt - 4 + 6) % 6];
   c04 = gear[(nualt - 5 + 6) % 6];
   c05 = gear[(nualt - 6 + 6) % 6];

   if (uinp) {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         uind[i][0] = c00 * udalt_00[i][0] + c01 * udalt_01[i][0] + c02 * udalt_02[i][0] + c03 * udalt_03[i][0]
            + c04 * udalt_04[i][0] + c05 * udalt_05[i][0];
         uind[i][1] = c00 * udalt_00[i][1] + c01 * udalt_01[i][1] + c02 * udalt_02[i][1] + c03 * udalt_03[i][1]
            + c04 * udalt_04[i][1] + c05 * udalt_05[i][1];
         uind[i][2] = c00 * udalt_00[i][2] + c01 * udalt_01[i][2] + c02 * udalt_02[i][2] + c03 * udalt_03[i][2]
            + c04 * udalt_04[i][2] + c05 * udalt_05[i][2];
         uinp[i][0] = c00 * upalt_00[i][0] + c01 * upalt_01[i][0] + c02 * upalt_02[i][0] + c03 * upalt_03[i][0]
            + c04 * upalt_04[i][0] + c05 * upalt_05[i][0];
         uinp[i][1] = c00 * upalt_00[i][1] + c01 * upalt_01[i][1] + c02 * upalt_02[i][1] + c03 * upalt_03[i][1]
            + c04 * upalt_04[i][1] + c05 * upalt_05[i][1];
         uinp[i][2] = c00 * upalt_00[i][2] + c01 * upalt_01[i][2] + c02 * upalt_02[i][2] + c03 * upalt_03[i][2]
            + c04 * upalt_04[i][2] + c05 * upalt_05[i][2];
      }
   } else {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         uind[i][0] = c00 * udalt_00[i][0] + c01 * udalt_01[i][0] + c02 * udalt_02[i][0] + c03 * udalt_03[i][0]
            + c04 * udalt_04[i][0] + c05 * udalt_05[i][0];
         uind[i][1] = c00 * udalt_00[i][1] + c01 * udalt_01[i][1] + c02 * udalt_02[i][1] + c03 * udalt_03[i][1]
            + c04 * udalt_04[i][1] + c05 * udalt_05[i][1];
         uind[i][2] = c00 * udalt_00[i][2] + c01 * udalt_01[i][2] + c02 * udalt_02[i][2] + c03 * udalt_03[i][2]
            + c04 * udalt_04[i][2] + c05 * udalt_05[i][2];
      }
   }
}

void ulspredSum_cu(real (*restrict uind)[3], real (*restrict uinp)[3])
{
   if (nualt < maxualt)
      return;

   if (polpred == UPred::ASPC) {
      launch_k1s(g::s0, n, ulspredSumASPC_cu,                                            //
         n, nualt, uind, uinp,                                                           //
         udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, //
         udalt_08, udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15, //
         upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07, //
         upalt_08, upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15);
   } else if (polpred == UPred::GEAR) {
      launch_k1s(g::s0, n, ulspredSumGEAR_cu,                        //
         n, nualt, uind, uinp,                                       //
         udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, //
         upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05);
   } else if (polpred == UPred::LSQR) {
      throwExceptionMissingFunction("ulspredSumLSQR_cu", __FILE__, __LINE__);
   }
}
}
