#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "seq/add.h"
#include "seq/launch.h"

namespace tinker {
__global__
void epolarEwaldRecipSelfEp_cu(int n, EnergyBuffer restrict ep, real f, //
   const real (*restrict fuind)[3], const real (*fphi)[20])
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = 0.5f * f * (fuind[i][0] * fphi[i][1] + fuind[i][1] * fphi[i][2] + fuind[i][2] * fphi[i][3]);
      atomic_add(e, ep, ithread);
   }
}

__global__
void epolarEwaldRecipSelfDep_cu(int n, real f,                                   //
   grad_prec* restrict depx, grad_prec* restrict depy, grad_prec* restrict depz, //
   const real (*restrict fmp)[10], const real (*restrict fphi)[20], const real (*restrict fuind)[3],
   const real (*restrict fuinp)[3], const real (*restrict fphid)[10], const real (*restrict fphip)[10],
   const real (*restrict fphidp)[20], //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   // data deriv1  / 2, 5,  8,  9, 11, 16, 18, 14, 15, 20 /
   // data deriv2  / 3, 8,  6, 10, 14, 12, 19, 16, 20, 17 /
   // data deriv3  / 4, 9, 10,  7, 15, 17, 13, 20, 18, 19 /
   constexpr int deriv1[10] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
   constexpr int deriv2[10] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
   constexpr int deriv3[10] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};

   if (fuinp) {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         real f1 = 0, f2 = 0, f3 = 0;
         for (int k = 0; k < 3; ++k) {
            int j1 = deriv1[k + 1];
            int j2 = deriv2[k + 1];
            int j3 = deriv3[k + 1];
            f1 += (fuind[i][k] + fuinp[i][k]) * fphi[i][j1];
            f2 += (fuind[i][k] + fuinp[i][k]) * fphi[i][j2];
            f3 += (fuind[i][k] + fuinp[i][k]) * fphi[i][j3];
            // if poltyp .eq. 'MUTUAL'
            f1 += fuind[i][k] * fphip[i][j1] + fuinp[i][k] * fphid[i][j1];
            f2 += fuind[i][k] * fphip[i][j2] + fuinp[i][k] * fphid[i][j2];
            f3 += fuind[i][k] * fphip[i][j3] + fuinp[i][k] * fphid[i][j3];
            // end if
         }
         for (int k = 0; k < 10; ++k) {
            f1 += fmp[i][k] * fphidp[i][deriv1[k]];
            f2 += fmp[i][k] * fphidp[i][deriv2[k]];
            f3 += fmp[i][k] * fphidp[i][deriv3[k]];
         }
         f1 *= 0.5f * nfft1;
         f2 *= 0.5f * nfft2;
         f3 *= 0.5f * nfft3;
         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;
         atomic_add(h1 * f, depx, i);
         atomic_add(h2 * f, depy, i);
         atomic_add(h3 * f, depz, i);
      }
   } else {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         real f1 = 0, f2 = 0, f3 = 0;
         for (int k = 0; k < 3; ++k) {
            int j1 = deriv1[k + 1];
            int j2 = deriv2[k + 1];
            int j3 = deriv3[k + 1];
            f1 += 2 * fuind[i][k] * fphi[i][j1];
            f2 += 2 * fuind[i][k] * fphi[i][j2];
            f3 += 2 * fuind[i][k] * fphi[i][j3];
            // if poltyp .eq. 'MUTUAL'
            f1 += 2 * fuind[i][k] * fphid[i][j1];
            f2 += 2 * fuind[i][k] * fphid[i][j2];
            f3 += 2 * fuind[i][k] * fphid[i][j3];
            // end if
         }
         for (int k = 0; k < 10; ++k) {
            f1 += fmp[i][k] * fphidp[i][deriv1[k]];
            f2 += fmp[i][k] * fphidp[i][deriv2[k]];
            f3 += fmp[i][k] * fphidp[i][deriv3[k]];
         }
         f1 *= 0.5f * nfft1;
         f2 *= 0.5f * nfft2;
         f3 *= 0.5f * nfft3;
         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;
         atomic_add(h1 * f, depx, i);
         atomic_add(h2 * f, depy, i);
         atomic_add(h3 * f, depz, i);
      }
   }
}

template <bool do_e, bool do_g, bool do_a>
__global__
void epolarEwaldRecipSelfEptrq_cu(int n, real term, real fterm,   //
   CountBuffer restrict nep, EnergyBuffer restrict ep,            //
   real* restrict trqx, real* restrict trqy, real* restrict trqz, //
   real* restrict pot,                                            //
   const real (*restrict rpole)[MPL_TOTAL], const real (*restrict cmp)[10], const real (*restrict gpu_uind)[3],
   const real (*restrict gpu_uinp)[3], const real (*restrict cphidp)[10])
{
   int ithread = ITHREAD;
   if (gpu_uinp) {
      for (int i = ithread; i < n; i += STRIDE) {
         real dix = rpole[i][MPL_PME_X];
         real diy = rpole[i][MPL_PME_Y];
         real diz = rpole[i][MPL_PME_Z];
         real uix = 0.5f * (gpu_uind[i][0] + gpu_uinp[i][0]);
         real uiy = 0.5f * (gpu_uind[i][1] + gpu_uinp[i][1]);
         real uiz = 0.5f * (gpu_uind[i][2] + gpu_uinp[i][2]);

         if CONSTEXPR (do_g) {
            real tep1 = cmp[i][3] * cphidp[i][2] - cmp[i][2] * cphidp[i][3] + 2 * (cmp[i][6] - cmp[i][5]) * cphidp[i][9]
               + cmp[i][8] * cphidp[i][7] + cmp[i][9] * cphidp[i][5] - cmp[i][7] * cphidp[i][8]
               - cmp[i][9] * cphidp[i][6];
            real tep2 = cmp[i][1] * cphidp[i][3] - cmp[i][3] * cphidp[i][1] + 2 * (cmp[i][4] - cmp[i][6]) * cphidp[i][8]
               + cmp[i][7] * cphidp[i][9] + cmp[i][8] * cphidp[i][6] - cmp[i][8] * cphidp[i][4]
               - cmp[i][9] * cphidp[i][7];
            real tep3 = cmp[i][2] * cphidp[i][1] - cmp[i][1] * cphidp[i][2] + 2 * (cmp[i][5] - cmp[i][4]) * cphidp[i][7]
               + cmp[i][7] * cphidp[i][4] + cmp[i][9] * cphidp[i][8] - cmp[i][7] * cphidp[i][5]
               - cmp[i][8] * cphidp[i][9];

            // self term

            tep1 += term * (diy * uiz - diz * uiy);
            tep2 += term * (diz * uix - dix * uiz);
            tep3 += term * (dix * uiy - diy * uix);

            atomic_add(tep1, trqx, i);
            atomic_add(tep2, trqy, i);
            atomic_add(tep3, trqz, i);

            // if (pot)
            //    atomic_add(cphidp[i][0], pot, i);
         }

         if CONSTEXPR (do_e) {
            uix = gpu_uind[i][0];
            uiy = gpu_uind[i][1];
            uiz = gpu_uind[i][2];
            real uii = dix * uix + diy * uiy + diz * uiz;
            atomic_add(fterm * uii, ep, ithread);
         }
         if CONSTEXPR (do_a)
            atomic_add(1, nep, ithread);
      }
   } else {
      for (int i = ithread; i < n; i += STRIDE) {
         real dix = rpole[i][MPL_PME_X];
         real diy = rpole[i][MPL_PME_Y];
         real diz = rpole[i][MPL_PME_Z];
         real uix = gpu_uind[i][0];
         real uiy = gpu_uind[i][1];
         real uiz = gpu_uind[i][2];

         if CONSTEXPR (do_g) {
            real tep1 = cmp[i][3] * cphidp[i][2] - cmp[i][2] * cphidp[i][3] + 2 * (cmp[i][6] - cmp[i][5]) * cphidp[i][9]
               + cmp[i][8] * cphidp[i][7] + cmp[i][9] * cphidp[i][5] - cmp[i][7] * cphidp[i][8]
               - cmp[i][9] * cphidp[i][6];
            real tep2 = cmp[i][1] * cphidp[i][3] - cmp[i][3] * cphidp[i][1] + 2 * (cmp[i][4] - cmp[i][6]) * cphidp[i][8]
               + cmp[i][7] * cphidp[i][9] + cmp[i][8] * cphidp[i][6] - cmp[i][8] * cphidp[i][4]
               - cmp[i][9] * cphidp[i][7];
            real tep3 = cmp[i][2] * cphidp[i][1] - cmp[i][1] * cphidp[i][2] + 2 * (cmp[i][5] - cmp[i][4]) * cphidp[i][7]
               + cmp[i][7] * cphidp[i][4] + cmp[i][9] * cphidp[i][8] - cmp[i][7] * cphidp[i][5]
               - cmp[i][8] * cphidp[i][9];

            // self term

            tep1 += term * (diy * uiz - diz * uiy);
            tep2 += term * (diz * uix - dix * uiz);
            tep3 += term * (dix * uiy - diy * uix);

            atomic_add(tep1, trqx, i);
            atomic_add(tep2, trqy, i);
            atomic_add(tep3, trqz, i);

            if (pot)
               atomic_add(cphidp[i][0], pot, i);
         }

         if CONSTEXPR (do_e) {
            uix = gpu_uind[i][0];
            uiy = gpu_uind[i][1];
            uiz = gpu_uind[i][2];
            real uii = dix * uix + diy * uiy + diz * uiz;
            atomic_add(fterm * uii, ep, ithread);
         }
         if CONSTEXPR (do_a)
            atomic_add(1, nep, ithread);
      }
   }
}

__global__
void epolarEwaldRecipSelfVirial_cu1(size_t size, VirialBuffer restrict vir_ep, const VirialBuffer restrict vir_m)
{
   for (size_t i = ITHREAD; i < size; i += STRIDE)
      vir_ep[0][i] -= vir_m[0][i];
}

__global__
void epolarEwaldRecipSelfVirial_cu2(int n, VirialBuffer restrict vir_ep, //
   const real (*restrict cmp)[10], const real (*restrict gpu_uind)[3], const real (*restrict gpu_uinp)[3],
   const real (*restrict fphid)[10], const real (*restrict fphip)[10], const real (*restrict cphi)[10],
   const real (*restrict cphidp)[10], //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   // frac_to_cart

   real ftc[3][3];
   ftc[0][0] = nfft1 * recipa.x;
   ftc[1][0] = nfft2 * recipb.x;
   ftc[2][0] = nfft3 * recipc.x;
   ftc[0][1] = nfft1 * recipa.y;
   ftc[1][1] = nfft2 * recipb.y;
   ftc[2][1] = nfft3 * recipc.y;
   ftc[0][2] = nfft1 * recipa.z;
   ftc[1][2] = nfft2 * recipb.z;
   ftc[2][2] = nfft3 * recipc.z;

   int ithread = ITHREAD;

   if (gpu_uinp) {
      for (int i = ithread; i < n; i += STRIDE) {
         real cphid[4], cphip[4];
         for (int j = 0; j < 3; ++j) {
            cphid[j + 1] = 0;
            cphip[j + 1] = 0;
            for (int k = 0; k < 3; ++k) {
               cphid[j + 1] += ftc[k][j] * fphid[i][k + 1];
               cphip[j + 1] += ftc[k][j] * fphip[i][k + 1];
            }
         }

         real vxx = 0;
         real vyy = 0;
         real vzz = 0;
         real vxy = 0;
         real vxz = 0;
         real vyz = 0;

         vxx = vxx - cmp[i][1] * cphidp[i][1] - 0.5f * ((gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][1]);
         vxy = vxy - 0.5f * (cphidp[i][1] * cmp[i][2] + cphidp[i][2] * cmp[i][1])
            - 0.25f * ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][1] + (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][2]);
         vxz = vxz - 0.5f * (cphidp[i][1] * cmp[i][3] + cphidp[i][3] * cmp[i][1])
            - 0.25f * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][1] + (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][3]);
         vyy = vyy - cmp[i][2] * cphidp[i][2] - 0.5f * ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][2]);
         vyz = vyz - 0.5f * (cphidp[i][2] * cmp[i][3] + cphidp[i][3] * cmp[i][2])
            - 0.25f * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][2] + (gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][3]);
         vzz = vzz - cmp[i][3] * cphidp[i][3] - 0.5f * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][3]);
         vxx = vxx - 2 * cmp[i][4] * cphidp[i][4] - cmp[i][7] * cphidp[i][7] - cmp[i][8] * cphidp[i][8];
         vxy = vxy - (cmp[i][4] + cmp[i][5]) * cphidp[i][7]
            - 0.5f * (cmp[i][7] * (cphidp[i][5] + cphidp[i][4]) + cmp[i][8] * cphidp[i][9] + cmp[i][9] * cphidp[i][8]);
         vxz = vxz - (cmp[i][4] + cmp[i][6]) * cphidp[i][8]
            - 0.5f * (cmp[i][8] * (cphidp[i][4] + cphidp[i][6]) + cmp[i][7] * cphidp[i][9] + cmp[i][9] * cphidp[i][7]);
         vyy = vyy - 2 * cmp[i][5] * cphidp[i][5] - cmp[i][7] * cphidp[i][7] - cmp[i][9] * cphidp[i][9];
         vyz = vyz - (cmp[i][5] + cmp[i][6]) * cphidp[i][9]
            - 0.5f * (cmp[i][9] * (cphidp[i][5] + cphidp[i][6]) + cmp[i][7] * cphidp[i][8] + cmp[i][8] * cphidp[i][7]);
         vzz = vzz - 2 * cmp[i][6] * cphidp[i][6] - cmp[i][8] * cphidp[i][8] - cmp[i][9] * cphidp[i][9];

         // if (poltyp .eq. 'MUTUAL')
         vxx = vxx - 0.5f * (cphid[1] * gpu_uinp[i][0] + cphip[1] * gpu_uind[i][0]);
         vxy = vxy
            - 0.25f
               * (cphid[1] * gpu_uinp[i][1] + cphip[1] * gpu_uind[i][1] + cphid[2] * gpu_uinp[i][0]
                  + cphip[2] * gpu_uind[i][0]);
         vxz = vxz
            - 0.25f
               * (cphid[1] * gpu_uinp[i][2] + cphip[1] * gpu_uind[i][2] + cphid[3] * gpu_uinp[i][0]
                  + cphip[3] * gpu_uind[i][0]);
         vyy = vyy - 0.5f * (cphid[2] * gpu_uinp[i][1] + cphip[2] * gpu_uind[i][1]);
         vyz = vyz
            - 0.25f
               * (cphid[2] * gpu_uinp[i][2] + cphip[2] * gpu_uind[i][2] + cphid[3] * gpu_uinp[i][1]
                  + cphip[3] * gpu_uind[i][1]);
         vzz = vzz - 0.5f * (cphid[3] * gpu_uinp[i][2] + cphip[3] * gpu_uind[i][2]);
         // end if

         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, ithread);
      }
   } else {
      for (int i = ithread; i < n; i += STRIDE) {
         real cphid[4];
         for (int j = 0; j < 3; ++j) {
            cphid[j + 1] = 0;
            for (int k = 0; k < 3; ++k) {
               cphid[j + 1] += ftc[k][j] * fphid[i][k + 1];
            }
         }

         real vxx = 0;
         real vyy = 0;
         real vzz = 0;
         real vxy = 0;
         real vxz = 0;
         real vyz = 0;

         vxx = vxx - cmp[i][1] * cphidp[i][1] - 0.5f * ((gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][1]);
         vxy = vxy - 0.5f * (cphidp[i][1] * cmp[i][2] + cphidp[i][2] * cmp[i][1])
            - 0.25f * ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][1] + (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][2]);
         vxz = vxz - 0.5f * (cphidp[i][1] * cmp[i][3] + cphidp[i][3] * cmp[i][1])
            - 0.25f * ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][1] + (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][3]);
         vyy = vyy - cmp[i][2] * cphidp[i][2] - 0.5f * ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][2]);
         vyz = vyz - 0.5f * (cphidp[i][2] * cmp[i][3] + cphidp[i][3] * cmp[i][2])
            - 0.25f * ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][2] + (gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][3]);
         vzz = vzz - cmp[i][3] * cphidp[i][3] - 0.5f * ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][3]);
         vxx = vxx - 2 * cmp[i][4] * cphidp[i][4] - cmp[i][7] * cphidp[i][7] - cmp[i][8] * cphidp[i][8];
         vxy = vxy - (cmp[i][4] + cmp[i][5]) * cphidp[i][7]
            - 0.5f * (cmp[i][7] * (cphidp[i][5] + cphidp[i][4]) + cmp[i][8] * cphidp[i][9] + cmp[i][9] * cphidp[i][8]);
         vxz = vxz - (cmp[i][4] + cmp[i][6]) * cphidp[i][8]
            - 0.5f * (cmp[i][8] * (cphidp[i][4] + cphidp[i][6]) + cmp[i][7] * cphidp[i][9] + cmp[i][9] * cphidp[i][7]);
         vyy = vyy - 2 * cmp[i][5] * cphidp[i][5] - cmp[i][7] * cphidp[i][7] - cmp[i][9] * cphidp[i][9];
         vyz = vyz - (cmp[i][5] + cmp[i][6]) * cphidp[i][9]
            - 0.5f * (cmp[i][9] * (cphidp[i][5] + cphidp[i][6]) + cmp[i][7] * cphidp[i][8] + cmp[i][8] * cphidp[i][7]);
         vzz = vzz - 2 * cmp[i][6] * cphidp[i][6] - cmp[i][8] * cphidp[i][8] - cmp[i][9] * cphidp[i][9];

         // if (poltyp .eq. 'MUTUAL')
         vxx = vxx - (cphid[1] * gpu_uind[i][0]);
         vxy = vxy - 0.25f * (2 * cphid[1] * gpu_uind[i][1] + 2 * cphid[2] * gpu_uind[i][0]);
         vxz = vxz - 0.25f * (2 * cphid[1] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][0]);
         vyy = vyy - cphid[2] * gpu_uind[i][1];
         vyz = vyz - 0.25f * (2 * cphid[2] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][1]);
         vzz = vzz - cphid[3] * gpu_uind[i][2];
         // end if

         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, ithread);
      }
   }
}

__global__
void epolarEwaldRecipSelfVirial_cu3(int n, real (*restrict cmp)[10], const real (*restrict gpu_uinp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      cmp[i][1] += gpu_uinp[i][0];
      cmp[i][2] += gpu_uinp[i][1];
      cmp[i][3] += gpu_uinp[i][2];
   }
}

__global__
void epolarEwaldRecipSelfVirial_cu4(int n, real (*restrict cmp)[10], const real (*restrict gpu_uind)[3],
   const real (*restrict gpu_uinp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      cmp[i][1] += (gpu_uind[i][0] - gpu_uinp[i][0]);
      cmp[i][2] += (gpu_uind[i][1] - gpu_uinp[i][1]);
      cmp[i][3] += (gpu_uind[i][2] - gpu_uinp[i][2]);
   }
}

__global__
void epolarEwaldRecipSelfVirial_cu5(int ntot, int nff, VirialBuffer restrict vir_ep, //
   real f, real volterm, real pterm, const PME* restrict d, const PME* restrict p,   //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < ntot; i += STRIDE) {
      if (i == 0) {
         continue;
      }

      // const real volterm = pi * box_volume;

      int k3 = i / nff;
      int j = i - k3 * nff;
      int k2 = j / nfft1;
      int k1 = j - k2 * nfft1;

      int r1 = (k1 < (nfft1 + 1) / 2) ? k1 : (k1 - nfft1);
      int r2 = (k2 < (nfft2 + 1) / 2) ? k2 : (k2 - nfft2);
      int r3 = (k3 < (nfft3 + 1) / 2) ? k3 : (k3 - nfft3);

      real h1 = recipa.x * r1 + recipb.x * r2 + recipc.x * r3;
      real h2 = recipa.y * r1 + recipb.y * r2 + recipc.y * r3;
      real h3 = recipa.z * r1 + recipb.z * r2 + recipc.z * r3;
      real hsq = h1 * h1 + h2 * h2 + h3 * h3;
      real term = -pterm * hsq;
      real expterm = 0;
      if (term > -50) {
         real denom = volterm * hsq * d->bsmod1[k1] * d->bsmod2[k2] * d->bsmod3[k3];
         expterm = REAL_EXP(term) / denom;
         if (box_shape == BoxShape::UNBOUND)
            expterm *= (1 - REAL_COS(pi * lvec1.x * REAL_SQRT(hsq)));
         else if (box_shape == BoxShape::OCT)
            if ((k1 + k2 + k3) & 1)
               expterm = 0; // end if ((k1 + k2 + k3) % 2 != 0)

         real struc2 = d->qgrid[2 * i] * p->qgrid[2 * i] + d->qgrid[2 * i + 1] * p->qgrid[2 * i + 1];
         real eterm = 0.5f * f * expterm * struc2;
         real vterm = (2 / hsq) * (1 - term) * eterm;

         real vxx = (h1 * h1 * vterm - eterm);
         real vxy = h1 * h2 * vterm;
         real vxz = h1 * h3 * vterm;
         real vyy = (h2 * h2 * vterm - eterm);
         real vyz = h2 * h3 * vterm;
         real vzz = (h3 * h3 * vterm - eterm);

         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, ithread);
      }
   }
}
}

namespace tinker {
template <class Ver>
static void epolarEwaldRecipSelf_cu1(const real (*gpu_uind)[3], const real (*gpu_uinp)[3])
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const PMEUnit pu = ppme_unit;
   const auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real aewald = st.aewald;

   const real f = electric / dielec;

   auto* fphid = fdip_phi1;
   auto* fphip = fdip_phi2;

   cuindToFuind(pu, gpu_uind, gpu_uinp, fuind, fuinp);
   if CONSTEXPR (do_e) {
      launch_k1b(g::s0, n, epolarEwaldRecipSelfEp_cu, n, ep, f, fuind, fphi);
   }
   gridUind(pu, fuind, fuinp);
   fftfront(pu);
   // TODO: store vs. recompute qfac
   pmeConv(pu);
   fftback(pu);
   fphiUind(pu, fphid, fphip, fphidp);

   // increment the dipole polarization gradient contributions

   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarEwaldRecipSelfDep_cu,  //
         n, f, depx, depy, depz,                        //
         fmp, fphi, fuind, fuinp, fphid, fphip, fphidp, //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
   }

   // set the potential to be the induced dipole average

   // see also subroutine eprecip1 in epolar1.f
   // do i = 1, npole
   //    do j = 1, 10
   //       fphidp(j,i) = 0.5d0 * fphidp(j,i)
   //    end do
   // end do
   // Notice that only 10 * n elements were scaled in the original code.
   darray::scale(g::q0, n, 0.5f * f, fphidp);
   fphiToCphi(pu, fphidp, cphidp);

   // recip and self torques

   real term = f * aewald * aewald * aewald * 4 / 3 / sqrtpi;
   real fterm = -2 * f * aewald * aewald * aewald / 3 / sqrtpi;
   launch_k1b(g::s0, n, epolarEwaldRecipSelfEptrq_cu<do_e, do_g, do_a>, //
      n, term, fterm, nep, ep, trqx, trqy, trqz, nullptr,               //
      rpole, cmp, gpu_uind, gpu_uinp, cphidp);

   // recip virial

   if CONSTEXPR (do_v) {
      auto size = bufferSize() * VirialBufferTraits::value;
      launch_k1s(g::s0, size, epolarEwaldRecipSelfVirial_cu1, size, vir_ep, vir_m);

      darray::scale(g::q0, n, f, cphi, fphid, fphip);

      launch_k1b(g::s0, n, epolarEwaldRecipSelfVirial_cu2,    //
         n, vir_ep,                                           //
         cmp, gpu_uind, gpu_uinp, fphid, fphip, cphi, cphidp, //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);

      // qgrip: pvu_qgrid
      const PMEUnit pvu = pvpme_unit;
      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu3, n, cmp, gpu_uinp);
      cmpToFmp(pvu, cmp, fmp);
      gridMpole(pvu, fmp);
      fftfront(pvu);

      // qgrid: pu_qgrid
      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu4, n, cmp, gpu_uind, gpu_uinp);
      cmpToFmp(pu, cmp, fmp);
      gridMpole(pu, fmp);
      fftfront(pu);

      const auto* d = pu.deviceptr();
      const auto* p = pvu.deviceptr();
      const int nff = nfft1 * nfft2;
      const int ntot = nfft1 * nfft2 * nfft3;
      real pterm = (pi / aewald) * (pi / aewald);
      real box_volume = boxVolume();
      real volterm = pi * box_volume;
      launch_k1b(g::s0, ntot, epolarEwaldRecipSelfVirial_cu5, //
         ntot, nff, vir_ep, f, volterm, pterm, d, p,          //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
   }
}

void epolarEwaldRecipSelf_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   if (vers == calc::v0) {
      epolarEwaldRecipSelf_cu1<calc::V0>(uind, uinp);
   } else if (vers == calc::v1) {
      epolarEwaldRecipSelf_cu1<calc::V1>(uind, uinp);
   } else if (vers == calc::v3) {
      epolarEwaldRecipSelf_cu1<calc::V3>(uind, uinp);
   } else if (vers == calc::v4) {
      epolarEwaldRecipSelf_cu1<calc::V4>(uind, uinp);
   } else if (vers == calc::v5) {
      epolarEwaldRecipSelf_cu1<calc::V5>(uind, uinp);
   } else if (vers == calc::v6) {
      epolarEwaldRecipSelf_cu1<calc::V6>(uind, uinp);
   }
}
}

namespace tinker {
template <class Ver>
static void epolarChgpenEwaldRecipSelf_cu1(const real (*gpu_uind)[3], bool use_cf)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const PMEUnit pu = ppme_unit;
   const auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real aewald = st.aewald;

   const real f = electric / dielec;

   auto* fphid = fdip_phi1;

   cuindToFuind(pu, gpu_uind, gpu_uind, fuind, fuind);
   if CONSTEXPR (do_e) {
      launch_k1b(g::s0, n, epolarEwaldRecipSelfEp_cu, n, ep, f, fuind, fphi);
   }
   gridUind(pu, fuind, fuind);
   fftfront(pu);
   // TODO: store vs. recompute qfac
   pmeConv(pu);
   fftback(pu);
   fphiUind(pu, fphid, fphid, fphidp);

   // increment the dipole polarization gradient contributions

   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarEwaldRecipSelfDep_cu,      //
         n, f, depx, depy, depz,                            //
         fmp, fphi, fuind, nullptr, fphid, nullptr, fphidp, //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
   }

   // set the potential to be the induced dipole average

   // see also subroutine eprecip1 in epolar1.f
   // do i = 1, npole
   //    do j = 1, 10
   //       fphidp(j,i) = 0.5d0 * fphidp(j,i)
   //    end do
   // end do
   // Notice that only 10 * n elements were scaled in the original code.
   darray::scale(g::q0, n, 0.5f * f, fphidp);
   fphiToCphi(pu, fphidp, cphidp);

   // recip and self torques

   real term = f * aewald * aewald * aewald * 4 / 3 / sqrtpi;
   real fterm = -2 * f * aewald * aewald * aewald / 3 / sqrtpi;
   launch_k1b(g::s0, n, epolarEwaldRecipSelfEptrq_cu<do_e, do_g, do_a>,    //
      n, term, fterm, nep, ep, trqx, trqy, trqz, (use_cf ? pot : nullptr), //
      rpole, cmp, gpu_uind, nullptr, cphidp);

   // recip virial

   if CONSTEXPR (do_v) {
      auto size = bufferSize() * VirialBufferTraits::value;
      launch_k1s(g::s0, size, epolarEwaldRecipSelfVirial_cu1, size, vir_ep, vir_m);

      darray::scale(g::q0, n, f, cphi, fphid);

      launch_k1b(g::s0, n, epolarEwaldRecipSelfVirial_cu2,     //
         n, vir_ep,                                            //
         cmp, gpu_uind, nullptr, fphid, nullptr, cphi, cphidp, //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);

      // qgrip: pvu_qgrid
      const PMEUnit pvu = pvpme_unit;
      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu3, n, cmp, gpu_uind);
      cmpToFmp(pvu, cmp, fmp);
      gridMpole(pvu, fmp);
      fftfront(pvu);

      // qgrid: pu_qgrid
      cmpToFmp(pu, cmp, fmp);
      gridMpole(pu, fmp);
      fftfront(pu);

      const auto* d = pu.deviceptr();
      const auto* p = pvu.deviceptr();
      const int nff = nfft1 * nfft2;
      const int ntot = nfft1 * nfft2 * nfft3;
      real pterm = (pi / aewald) * (pi / aewald);
      real box_volume = boxVolume();
      real volterm = pi * box_volume;
      launch_k1b(g::s0, ntot, epolarEwaldRecipSelfVirial_cu5, //
         ntot, nff, vir_ep, f, volterm, pterm, d, p,          //
         nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
   }
}

void epolarChgpenEwaldRecipSelf_cu(int vers, int use_cf, const real (*uind)[3])
{
   if (vers == calc::v0) {
      if (use_cf)
         assert(false && "CFLX must compute gradient.");
      else
         epolarChgpenEwaldRecipSelf_cu1<calc::V0>(uind, false);
   } else if (vers == calc::v1) {
      epolarChgpenEwaldRecipSelf_cu1<calc::V1>(uind, use_cf);
   } else if (vers == calc::v3) {
      if (use_cf)
         assert(false && "CFLX must compute gradient.");
      else
         epolarChgpenEwaldRecipSelf_cu1<calc::V3>(uind, false);
   } else if (vers == calc::v4) {
      epolarChgpenEwaldRecipSelf_cu1<calc::V4>(uind, use_cf);
   } else if (vers == calc::v5) {
      epolarChgpenEwaldRecipSelf_cu1<calc::V5>(uind, use_cf);
   } else if (vers == calc::v6) {
      epolarChgpenEwaldRecipSelf_cu1<calc::V6>(uind, use_cf);
   }
}
}
