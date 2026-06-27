#pragma once
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/seq.h"

namespace tinker {
// ufield, PCG
SEQ_ROUTINE
inline void dmpthole2_a(real r, real pgamma, real pdi, real pdk, real& restrict scale3, real& restrict scale5)
{
   real damp = pdi * pdk;
   real ratio = r * REAL_RECIP(damp);
   damp = (damp == 0 ? ((real)-1.0e16) : -pgamma * ratio * ratio * ratio);
   real expdamp = REAL_EXP(damp);
   scale3 = 1 - expdamp;
   scale5 = 1 - expdamp * (1 - damp);
}

SEQ_ROUTINE
inline void damp_thole2(real r, real pdi, real pti, real pdk, real ptk, real& restrict scale3, real& restrict scale5)
{
   real pgamma = REAL_MIN(pti, ptk);
   dmpthole2_a(r, pgamma, pdi, pdk, scale3, scale5);
}

// dfield
SEQ_ROUTINE
inline void dmpthole3_a(real r, real pgamma, real pdi, real pdk, real& restrict scale3, real& restrict scale5, real& restrict scale7)
{
   real damp = pdi * pdk;
   real ratio = r * REAL_RECIP(damp);
   damp = (damp == 0 ? ((real)-1.0e16) : -pgamma * ratio * ratio * ratio);
   real expdamp = REAL_EXP(damp);
   scale3 = 1 - expdamp;
   scale5 = 1 - expdamp * (1 - damp);
   scale7 = 1 - expdamp * (1 - damp + (real)0.6 * damp * damp);
}

SEQ_ROUTINE
inline void damp_thole3(real r, real pdi, real pti, real pdk, real ptk, real& restrict scale3, real& restrict scale5, real& restrict scale7)
{
   real pgamma = REAL_MIN(pti, ptk);
   dmpthole3_a(r, pgamma, pdi, pdk, scale3, scale5, scale7);
}

SEQ_ROUTINE
inline void dmpthole3g_a(real r,
                         real rr2,
                         real xr,
                         real yr,
                         real zr,
                         real pgamma,
                         real pdi,
                         real pdk,
                         real& restrict scale31,
                         real& restrict scale51,
                         real& restrict scale71,
                         real& restrict rc31,
                         real& restrict rc32,
                         real& restrict rc33,
                         real& restrict rc51,
                         real& restrict rc52,
                         real& restrict rc53,
                         real& restrict rc71,
                         real& restrict rc72,
                         real& restrict rc73)
{
   real damp = pdi * pdk;
   real ratio = r * REAL_RECIP(damp);
   damp = (damp == 0 ? ((real)1.0e16) : pgamma * ratio * ratio * ratio);
   scale31 = REAL_EXP(-damp);
   scale51 = scale31 * (1 + damp);
   scale71 = scale31 * (1 + damp + (real)0.6 * damp * damp);
   real temp3 = 3 * damp * scale31 * rr2;
   real temp5 = damp;
   real temp7 = (real)-0.2 + (real)0.6 * damp;
   rc31 = xr * temp3;
   rc32 = yr * temp3;
   rc33 = zr * temp3;
   rc51 = rc31 * temp5;
   rc52 = rc32 * temp5;
   rc53 = rc33 * temp5;
   rc71 = rc51 * temp7;
   rc72 = rc52 * temp7;
   rc73 = rc53 * temp7;
}

SEQ_ROUTINE
inline void damp_thole3g(real r,
                         real rr2,
                         real xr,
                         real yr,
                         real zr,
                         real pdi,
                         real pti,
                         real pdk,
                         real ptk,
                         real& restrict scale31,
                         real& restrict scale51,
                         real& restrict scale71,
                         real& restrict rc31,
                         real& restrict rc32,
                         real& restrict rc33,
                         real& restrict rc51,
                         real& restrict rc52,
                         real& restrict rc53,
                         real& restrict rc71,
                         real& restrict rc72,
                         real& restrict rc73)
{
   real pgamma = REAL_MIN(pti, ptk);
   dmpthole3g_a(r, rr2, xr, yr, zr, pgamma, pdi, pdk, scale31, scale51, scale71, rc31, rc32, rc33, rc51, rc52, rc53, rc71, rc72, rc73);
}

SEQ_ROUTINE
inline void dmpthole4_a(real r, real pgamma, real pdi, real pdk, real& restrict ex3, real& restrict ex5, real& restrict ex7, real& restrict ex9)
{
   real damp = pdi * pdk;
   real ratio = r * REAL_RECIP(damp);
   damp = (damp == 0 ? ((real)1.0e16) : pgamma * ratio * ratio * ratio);
   ex3 = REAL_EXP(-damp);
   ex5 = ex3 * (1 + damp);
   ex7 = ex3 * (1 + damp + (real)0.6 * damp * damp);
   constexpr real coef1 = ((real)18) / ((real)35);
   constexpr real coef2 = ((real)9) / ((real)35);
   ex9 = ex3 * (1 + damp * (1 + damp * (coef1 + coef2 * damp)));
}

SEQ_ROUTINE
inline void damp_thole4(real r, real pdi, real pti, real pdk, real ptk, real& restrict ex3, real& restrict ex5, real& restrict ex7, real& restrict ex9)
{
   real pgamma = REAL_MIN(pti, ptk);
   dmpthole4_a(r, pgamma, pdi, pdk, ex3, ex5, ex7, ex9);
}

#pragma acc routine seq
template <int order>
SEQ_CUDA
inline void damp_ewald(real* restrict bn, real r, real rinv, real rr2, real aewald)
{
   real ralpha = aewald * r;
   bn[0] = REAL_ERFC(ralpha) * rinv;
   real alsq2 = 2 * aewald * aewald;
   real alsq2n = REAL_RECIP(sqrtpi * aewald);
   real exp2a = REAL_EXP(-ralpha * ralpha);
   #pragma unroll
   for (int j = 1; j < order; ++j) {
      alsq2n *= alsq2;
      bn[j] = ((j + j - 1) * bn[j - 1] + alsq2n * exp2a) * rr2;
   }
}
}
