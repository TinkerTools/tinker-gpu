#pragma once
#include "ff/energybuffer.h"
#include "ff/precision.h"

namespace tinker {
/// \addtogroup pme
/// \{
///
/// Per-site pieces of the multipole PME reciprocal sum.
///
/// Every expression here is **bilinear** in its two array arguments -- each term is exactly one
/// factor from the first array times one from the second. That is what makes the lambda derivative
/// cheap and safe to form: for any of these H,
/// \f[ \frac{d}{d\lambda} H(cmp, cphi) = H(dlcmp, cphi) + H(cmp, dlcphi) \f]
/// so the dlmda kernel calls the same helper twice and adds, instead of transcribing the product
/// rule term by term.

/// Reciprocal contraction for site \c i: the energy-like sum and the three fractional force
/// components. Bilinear in (\c fmp, \c fphi).
template <bool DO_E, bool DO_G>
__device__
inline void emrecipEnergyForceAtomI(int i, const real (*restrict fmp)[10], const real (*restrict fphi)[20],
   real& e, real& f1, real& f2, real& f3)
{
   constexpr int deriv1[] = {2, 5, 8, 9, 11, 16, 18, 14, 15, 20};
   constexpr int deriv2[] = {3, 8, 6, 10, 14, 12, 19, 16, 20, 17};
   constexpr int deriv3[] = {4, 9, 10, 7, 15, 17, 13, 20, 18, 19};

   if CONSTEXPR (DO_E)
      e = 0;
   if CONSTEXPR (DO_G) {
      f1 = 0;
      f2 = 0;
      f3 = 0;
   }
   for (int k = 0; k < 10; ++k) {
      if CONSTEXPR (DO_E)
         e += fmp[i][k] * fphi[i][k];
      if CONSTEXPR (DO_G) {
         f1 += fmp[i][k] * fphi[i][deriv1[k] - 1];
         f2 += fmp[i][k] * fphi[i][deriv2[k] - 1];
         f3 += fmp[i][k] * fphi[i][deriv3[k] - 1];
      }
   }
}

/// Reciprocal torque on site \c i. Bilinear in (\c cmp, \c cphi). Not scaled by \c f.
__device__
inline void emrecipTorqueAtomI(int i, const real (*restrict cmp)[10], const real (*restrict cphi)[10],
   real (&tem)[3])
{
   tem[0] = cmp[i][3] * cphi[i][2] - cmp[i][2] * cphi[i][3] + 2 * (cmp[i][6] - cmp[i][5]) * cphi[i][9]
      + cmp[i][8] * cphi[i][7] + cmp[i][9] * cphi[i][5] - cmp[i][7] * cphi[i][8] - cmp[i][9] * cphi[i][6];
   tem[1] = cmp[i][1] * cphi[i][3] - cmp[i][3] * cphi[i][1] + 2 * (cmp[i][4] - cmp[i][6]) * cphi[i][8]
      + cmp[i][7] * cphi[i][9] + cmp[i][8] * cphi[i][6] - cmp[i][8] * cphi[i][4] - cmp[i][9] * cphi[i][7];
   tem[2] = cmp[i][2] * cphi[i][1] - cmp[i][1] * cphi[i][2] + 2 * (cmp[i][5] - cmp[i][4]) * cphi[i][7]
      + cmp[i][7] * cphi[i][4] + cmp[i][9] * cphi[i][8] - cmp[i][7] * cphi[i][5] - cmp[i][8] * cphi[i][9];
}

/// Reciprocal virial contribution of site \c i, ordered {xx, xy, xz, yy, yz, zz}.
/// Bilinear in (\c cmp, \c cphi). Not scaled by \c f.
__device__
inline void emrecipVirialAtomI(int i, const real (*restrict cmp)[10], const real (*restrict cphi)[10],
   real (&v)[6])
{
   v[0] = -cmp[i][1] * cphi[i][1] - 2 * cmp[i][4] * cphi[i][4] - cmp[i][7] * cphi[i][7] - cmp[i][8] * cphi[i][8];
   v[1] = -0.5f * (cmp[i][2] * cphi[i][1] + cmp[i][1] * cphi[i][2]) - (cmp[i][4] + cmp[i][5]) * cphi[i][7]
      - 0.5f * cmp[i][7] * (cphi[i][4] + cphi[i][5]) - 0.5f * (cmp[i][8] * cphi[i][9] + cmp[i][9] * cphi[i][8]);
   v[2] = -0.5f * (cmp[i][3] * cphi[i][1] + cmp[i][1] * cphi[i][3]) - (cmp[i][4] + cmp[i][6]) * cphi[i][8]
      - 0.5f * cmp[i][8] * (cphi[i][4] + cphi[i][6]) - 0.5f * (cmp[i][7] * cphi[i][9] + cmp[i][9] * cphi[i][7]);
   v[3] = -cmp[i][2] * cphi[i][2] - 2 * cmp[i][5] * cphi[i][5] - cmp[i][7] * cphi[i][7] - cmp[i][9] * cphi[i][9];
   v[4] = -0.5f * (cmp[i][3] * cphi[i][2] + cmp[i][2] * cphi[i][3]) - (cmp[i][5] + cmp[i][6]) * cphi[i][9]
      - 0.5f * cmp[i][9] * (cphi[i][5] + cphi[i][6]) - 0.5f * (cmp[i][7] * cphi[i][8] + cmp[i][8] * cphi[i][7]);
   v[5] = -cmp[i][3] * cphi[i][3] - 2 * cmp[i][6] * cphi[i][6] - cmp[i][8] * cphi[i][8] - cmp[i][9] * cphi[i][9];
}
/// \}
}
