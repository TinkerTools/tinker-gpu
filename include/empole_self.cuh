#pragma once
#include "add.h"
#include "md.h"


TINKER_NAMESPACE_BEGIN
template <int USE>
__global__
void empole_self_cu(size_t bufsize, count_buffer restrict nem,
                    energy_buffer restrict em, const real (*restrict rpole)[10],
                    int n, real f, real aewald)
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_a = USE & calc::analyz;
   static_assert(do_e, "");


   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);


   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
      real ci = rpole[i][mpl_pme_0];
      real dix = rpole[i][mpl_pme_x];
      real diy = rpole[i][mpl_pme_y];
      real diz = rpole[i][mpl_pme_z];
      real qixx = rpole[i][mpl_pme_xx];
      real qixy = rpole[i][mpl_pme_xy];
      real qixz = rpole[i][mpl_pme_xz];
      real qiyy = rpole[i][mpl_pme_yy];
      real qiyz = rpole[i][mpl_pme_yz];
      real qizz = rpole[i][mpl_pme_zz];


      real cii = ci * ci;
      real dii = dix * dix + diy * diy + diz * diz;
      real qii = 2 * (qixy * qixy + qixz * qixz + qiyz * qiyz) + qixx * qixx +
         qiyy * qiyy + qizz * qizz;


      int offset = i & (bufsize - 1);
      real e = fterm *
         (cii + aewald_sq_2 * (dii / 3 + 2 * aewald_sq_2 * qii * (real)0.2));
      atomic_add(e, em, offset);
      if CONSTEXPR (do_a)
         atomic_add(1, nem, offset);
   }
}
TINKER_NAMESPACE_END
