#pragma once
#include "ff/dlmda.h"
#include "seq/add.h"
#include "seq/seq.h"

// lambda derivative helpers shared by the pair kernels of every term
namespace tinker {
/// \ingroup ff
/// Adds the lambda derivative of a pair force, \c dldr times the separation
/// `(xr, yr, zr) = r_i - r_k`, to the lambda derivatives of the gradients of
/// sites i and k if \c DO_G and of the virial if \c DO_V (dlambda.f:dlcpair,
/// dlcvir).
#pragma acc routine seq
template <bool DO_G, bool DO_V, class V>
SEQ_CUDA
inline void pair_dlforce(real dldr,
                         real xr,
                         real yr,
                         real zr,
                         real& fix,
                         real& fiy,
                         real& fiz,
                         real& fkx,
                         real& fky,
                         real& fkz,
                         V& vxx,
                         V& vyx,
                         V& vzx,
                         V& vyy,
                         V& vzy,
                         V& vzz)
{
   real dldx = dldr * xr;
   real dldy = dldr * yr;
   real dldz = dldr * zr;
   if CONSTEXPR (DO_G) {
      fix += dldx;
      fiy += dldy;
      fiz += dldz;
      fkx -= dldx;
      fky -= dldy;
      fkz -= dldz;
   }
   if CONSTEXPR (DO_V) {
      vxx += floatTo<V>(xr * dldx);
      vyx += floatTo<V>(yr * dldx);
      vzx += floatTo<V>(zr * dldx);
      vyy += floatTo<V>(yr * dldy);
      vzy += floatTo<V>(zr * dldy);
      vzz += floatTo<V>(zr * dldz);
   }
}

/// \ingroup ff
/// The scale of the group pair \c gi, \c gk with its main lambda derivatives
/// (GrpPairScale).
struct GrpPairLmda
{
   real sc, dl, d2l;
};

SEQ_ROUTINE
inline GrpPairLmda grpPairLmda(const GrpPairScale& ps, int gi, int gk)
{
   return {ps.sc[gi][gk], ps.dl[gi][gk], ps.d2l[gi][gk]};
}

/// \ingroup ff
/// Whether a group pair \c w has a lambda derivative to take: its slope, or
/// with \c DL2 its curvature. A first derivative version never reads the
/// curvature, which the kernels would otherwise load from local memory.
#pragma acc routine seq
template <bool DL2>
SEQ_CUDA
inline bool grpPairLmdaActive(const GrpPairLmda& w)
{
   return w.dl != 0 or (DL2 and w.d2l != 0);
}

/// \ingroup ff
/// Adds the lambda derivatives of a pair energy \c e taken before its group
/// pair scale \c w, `w.dl * e` and `w.d2l * e`, to \c dl1 if \c DL1 and to
/// \c dl2 if \c DL2 (dlambda.f:dlcscale). The caller skips the pair unless
/// grpPairLmdaActive(), tested in the condition of its own branch: a helper
/// that returns the test is kept as a value and costs a select.
#pragma acc routine seq
template <bool DL1, bool DL2, class E>
SEQ_CUDA
inline void pair_grpscale_dl(const GrpPairLmda& w, const real& e, E& dl1, E& dl2)
{
   if CONSTEXPR (DL1)
      dl1 += floatTo<E>(w.dl * e);
   if CONSTEXPR (DL2)
      dl2 += floatTo<E>(w.d2l * e);
}
}
