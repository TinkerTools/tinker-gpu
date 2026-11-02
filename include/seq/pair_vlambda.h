#pragma once
#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
/**
 * \ingroup vdw
 * \brief Whether a pair is scaled by the van der Waals lambda. A mutated pair
 * at lambda one still has a lambda derivative, so this is not vlambda < 1.
 */
SEQ_ROUTINE
inline bool pair_mutik(Vdw vcouple, int imut, int kmut)
{
   // without branches, since the kernels ask it of every pair
   return ((vcouple == Vdw::DECOUPLE) & (imut != kmut)) | ((vcouple == Vdw::ANNIHILATE) & ((imut | kmut) != 0));
}

/**
 * \ingroup vdw
 * \brief The main lambda derivatives of a van der Waals pair: of its energy,
 * \c dedl and \c d2edl2, and of the distance derivative of its energy, \c dlde.
 */
struct PairVdwLambda
{
   real dedl;
   real d2edl2;
   real dlde;
};

/**
 * \ingroup vdw
 * \brief The class of a pair of ligand groups from the table that
 * vdwMutclsTable() packs (dlambda.f:grppcls): -1 for a pair of the two ligands
 * of a relative calculation, which never interact, 0 for a pair at full
 * weight, and otherwise the ligand group whose soft core scales the pair.
 * Outside a relative calculation the groups are the 0/1 mutation flags.
 */
SEQ_ROUTINE
inline int pair_mutcls(int table, int imutg, int kmutg)
{
   return ((table >> (2 * (3 * imutg + kmutg))) & 3) - 1;
}

/**
 * \ingroup vdw
 * \brief The soft core of ligand group 2 if \c two, else of group 1, chosen
 * field by field, so that the kernel parameters are not copied to local memory
 * to be chosen by address.
 */
SEQ_ROUTINE
inline SoftCoreLmda pair_softcore(bool two, const SoftCoreLmda& s1, const SoftCoreLmda& s2)
{
   return {two ? s2.p0 : s1.p0, two ? s2.sc : s1.sc, two ? s2.a1 : s1.a1, two ? s2.b1 : s1.b1, two ? s2.a2 : s1.a2, two ? s2.b2 : s1.b2, two ? s2.c2 : s1.c2};
}

/**
 * \ingroup vdw
 * \brief The soft core of a pair of class \c mc (pair_mutcls): the plain
 * pair, a scale of one without a shift, unless it follows lambda, and then
 * that of ligand group 2 for class 2 of a relative calculation (REL), else of
 * group 1.
 */
#pragma acc routine seq
template <bool REL>
SEQ_CUDA
SoftCoreLmda pair_vdw_softcore(int mc, const SoftCoreLmda& s1, const SoftCoreLmda& s2)
{
   SoftCoreLmda lp = {1, 0, 0, 0, 0, 0, 0};
   if (mc > 0)
      lp = REL ? pair_softcore(mc == 2, s1, s2) : s1;
   return lp;
}

/**
 * \ingroup vdw
 * \brief The main lambda derivatives of a soft core pair \f$ p L T(s) \f$ from
 * the coefficients of its soft core \c lp (SoftCoreLmda): of its energy from
 * \c tt, the pair function T, and its shift derivatives \c ts and \c tss, and
 * of its distance derivative from \c tr and \c trs, the distance derivatives
 * of T and of its shift derivative, times \c rfac (eljsc.f:eljsc4,
 * ehalsc.f:ehalsc4). The force takes the energy derivative for its taper.
 */
#pragma acc routine seq
template <bool DL1, bool DL2, bool DLDE>
SEQ_CUDA
void pair_softcore_dl(const SoftCoreLmda& lp, real pref, real tt, real ts, real tss, real tr, real trs, real rfac, PairVdwLambda* restrict dl)
{
   if CONSTEXPR (DL1 or DLDE)
      dl->dedl = pref * (lp.a1 * tt + lp.b1 * ts);
   if CONSTEXPR (DL2)
      dl->d2edl2 = pref * (lp.a2 * tt + lp.b2 * ts + lp.c2 * tss);
   if CONSTEXPR (DLDE)
      dl->dlde = pref * (lp.a1 * tr + lp.b1 * trs) * rfac;
}

/**
 * \ingroup vdw
 * \brief Tapers a van der Waals pair, \c ev and its distance derivative
 * \c dev, and its main lambda derivatives, near the cutoff (elj1.f, ehal1.f).
 */
#pragma acc routine seq
template <bool DO_G, bool DL1, bool DL2, bool DLDE>
SEQ_CUDA
void pair_vdw_taper(real r, real evcut, real evoff, real& restrict ev, real& restrict dev, PairVdwLambda* restrict dl)
{
   if (r > evcut) {
      real taper, dtaper;
      switchTaper5<DO_G or DLDE>(r, evcut, evoff, taper, dtaper);
      if CONSTEXPR (DO_G)
         dev = ev * dtaper + dev * taper;
      ev = ev * taper;
      if CONSTEXPR (DLDE)
         dl->dlde = dl->dedl * dtaper + dl->dlde * taper;
      if CONSTEXPR (DL2)
         dl->d2edl2 *= taper;
      if CONSTEXPR (DL1)
         dl->dedl *= taper;
   }
}

/// \ingroup vdw
/// Adds the lambda derivative of a pair force, \c dldr times the separation
/// `(xr, yr, zr) = r_i - r_k`, to the lambda derivatives of the gradients of
/// sites i and k if \c DO_G and of the virial if \c DO_V (dlambda.f:dlcpair,
/// dlcvir).
#pragma acc routine seq
template <bool DO_G, bool DO_V, class V>
SEQ_CUDA
void pair_dlforce(real dldr,
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
void pair_grpscale_dl(const GrpPairLmda& w, const real& e, E& dl1, E& dl2)
{
   if CONSTEXPR (DL1)
      dl1 += floatTo<E>(w.dl * e);
   if CONSTEXPR (DL2)
      dl2 += floatTo<E>(w.d2l * e);
}
}
