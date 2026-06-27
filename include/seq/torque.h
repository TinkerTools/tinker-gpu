#pragma once
#include "ff/precision.h"
#include "seq/seq.h"

namespace tinker {
SEQ_ROUTINE
inline real torqueDot(const real* restrict a, const real* restrict b)
{
   return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

SEQ_ROUTINE
inline void torqueCross(real* restrict ans, const real* restrict u, const real* restrict v)
{
   ans[0] = u[1] * v[2] - u[2] * v[1];
   ans[1] = u[2] * v[0] - u[0] * v[2];
   ans[2] = u[0] * v[1] - u[1] * v[0];
}

SEQ_ROUTINE
inline void torqueNormal(real* restrict a, real _1_na)
{
   a[0] *= _1_na;
   a[1] *= _1_na;
   a[2] *= _1_na;
}
}
