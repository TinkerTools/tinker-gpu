#pragma once
#include "tool/macro.h"

/// \ingroup prec
/// \def TINKER_DETERMINISTIC_FORCE
/// Logical macro for the underlying type of energy gradients.
///    - If \c true, always uses fixed-point arithmetic to accumulate energy
///    gradients, regardless of the underlying type of each individual component.
///    - If \c false, use type \c real, which can either be \c float or \c double
///    based on the precision macro.
///
/// \see TINKER_DOUBLE_PRECISION
/// \see TINKER_MIXED_PRECISION
/// \see TINKER_SINGLE_PRECISION
///
/// In general, evaluating energy, forces, etc. twice, we don't expect to get
/// two identical answers, but we may not care as much because the difference
/// is usually negligible.
/// [See [Why is cos(x) !=
/// cos(y)?](https://isocpp.org/wiki/faq/newbie#floating-point-arith2)]
/// Whereas in MD, two simulations with the same initial configurations can
/// easily diverge due to the accumulated difference. If, for whatever reason,
/// you are willing to elongate the process of the inevitable divergence at the
/// cost of slightly slower simulation speed, a more "deterministic" force (using
/// fixed-point arithmetic) can help.
///
/// To accumulate floating-point values via fixed-point arithmetic, we first
/// scale the values by a large integer then sum only the integer part.
/// \code{.cpp}
/// // fixed sum = 0;
/// // FLT is float or double
/// FLT val2 = val * 0x100000000ull;
/// sum += (fixed)((long long)val2);
/// \endcode
/// To get the floating-point sum, we have to cast the integer sum back to
/// floating-point number, then divide it by the same large integer.
/// \code{.cpp}
/// FLT val2 = (FLT)((long long)sum);
/// FLT answer = val2 / 0x100000000ull;
/// \endcode
#ifndef TINKER_DETERMINISTIC_FORCE
#if TINKER_DOUBLE_PRECISION
#define TINKER_DETERMINISTIC_FORCE 0
#else
#define TINKER_DETERMINISTIC_FORCE 1
#endif
#endif

namespace tinker {
/// \addtogroup prec
/// \{

/// \typedef fixed
/// 64-bit unsigned integer type for fixed-point arithmetic.
///
/// \typedef real
/// Floating-point type with lower precision (not higher than #mixed).
/// \see TINKER_MIXED_PRECISION
///
/// \typedef mixed
/// Floating-point type with higher precision (not lower than #real).
/// \see TINKER_MIXED_PRECISION
///
/// \def TINKER_REAL_SIZE
/// Number of bytes of the type \c real.
/// \def TINKER_MIXED_SIZE
/// Number of bytes of the type \c mixed.
typedef unsigned long long fixed;
static_assert(sizeof(fixed) == 8, "");

#if TINKER_DOUBLE_PRECISION
#define TINKER_REAL_SIZE  8
#define TINKER_MIXED_SIZE 8
typedef double real;
typedef double mixed;
#endif
#if TINKER_MIXED_PRECISION
#define TINKER_REAL_SIZE  4
#define TINKER_MIXED_SIZE 8
typedef float real;
typedef double mixed;
#endif
#if TINKER_SINGLE_PRECISION
#define TINKER_REAL_SIZE  4
#define TINKER_MIXED_SIZE 4
typedef float real;
typedef float mixed;
#endif

typedef mixed time_prec;   ///< Floating-point type for time.
typedef mixed T_prec;      ///< Floating-point type for temperature.
typedef mixed vel_prec;    ///< Floating-point type for velocities.
typedef mixed pos_prec;    ///< Floating-point type for coordinates.
typedef real e_prec;       ///< Floating-point type for the pairwise energy components.
typedef real v_prec;       ///< Floating-point type for the pairwise virial components.
typedef real g_prec;       ///< Floating-point type for the pairwise gradient components.
typedef mixed energy_prec; ///< Floating-point type for total energies.
typedef mixed virial_prec; ///< Floating-point type for total virials.
/// \typedef grad_prec
/// Floating-point or fixed-point type for the total gradients.
/// \see TINKER_DETERMINISTIC_FORCE
#if TINKER_DETERMINISTIC_FORCE
typedef fixed grad_prec;
#else
typedef real grad_prec;
#endif
/// \}
}
