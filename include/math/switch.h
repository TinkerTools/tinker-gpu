#pragma once
#include "ff/precision.h"
#include "seq/seq.h"

namespace tinker {
/// \ingroup math
/// \brief Smooth function `F: [cut,off]->[1,0]`.
///
/// Derived from the 5th order `smoothstep` function (-S2(x) + 1):
/// \f[ S_2: [0,1]\rightarrow[0,1] \f]
/// \f[ S_2(x) = 6 x^5 - 15 x^4 + 10 x^3 \f]
///
/// \param[in]  rik     Distance.
/// \param[in]  cut     Distance at which switching of the potential begins.
/// \param[out] off     Distance at which the potential energy goes to zero.
/// \param[out] taper   F value.
/// \param[out] dtaper  dF/dx value.
/// \tparam DO_DTAPER   If \c false, \c dtaper will not be calculated and its
///                     original value will not be altered.
#pragma acc routine seq
template <int DO_DTAPER>
SEQ_CUDA
void switchTaper5(real rik, real cut, real off, real& restrict taper, real& restrict dtaper)
{
   real rinv = 1 / (cut - off);
   real x = (rik - off) * rinv;
   real x2 = x * x;
   real x3 = x2 * x;
   taper = x3 * (6 * x2 - 15 * x + 10);
   if CONSTEXPR (DO_DTAPER)
      dtaper = 30 * (x * (1 - x)) * (x * (1 - x)) * rinv;
}
}
