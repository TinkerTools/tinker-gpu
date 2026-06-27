#pragma once
#include <cstddef>

// constexpr functions
namespace tinker {
/// \ingroup math
/// \brief Return true if and only if `val` is a power of 2.
constexpr bool isPow2(size_t val)
{
   return (val != 0) && ((val & (val - 1)) == 0);
}

/// \ingroup math
/// \brief Integer base 2 power.
constexpr int pow2(int val)
{
   return 1 << val;
}

/// \ingroup math
/// \brief Long long integer base 2 power.
constexpr long long pow2ll(int val)
{
   return 1ll << val;
}

/// \ingroup math
/// \brief The base-2 logarithm of `val`.
/// \param val  Must be greater than 0.
/// \note Returns 0 if `val` is 0.
constexpr int floorLog2_constexpr(size_t val)
{
   return val < 2 ? 0 : 1 + floorLog2_constexpr(val >> 1);
}

/// \ingroup math
/// \brief A power of 2 that is less than or equal to `val`.
/// \param val  Must be greater than 0.
constexpr size_t pow2Le(size_t val)
{
   return 1ull << floorLog2_constexpr(val);
}

/// \ingroup math
/// \brief A power of 2 that is greater than or equal to `val`.
constexpr size_t pow2Ge(size_t val)
{
   return val <= 1 ? 1 : (1ull << (1 + floorLog2_constexpr(val - 1)));
}
}

// inline functions
namespace tinker {
/// \ingroup math
/// \brief Non-constexpr `floorLog2(int)`.
inline int floorLog2(int val)
{
   if (val == 0)
      return 0;
   int c = __builtin_clz(val);
   return 8 * sizeof(int) - 1 - c;
}

/// \ingroup math
/// \brief Non-constexpr `floorLog2(long long)`.
inline int floorLog2(long long val)
{
   if (val == 0)
      return 0;
   int c = __builtin_clzll(val);
   return 8 * sizeof(long long) - 1 - c;
}

/// \ingroup math
/// \brief Non-constexpr log base 2 ceiling.
inline int ceilLog2(int val)
{
   return floorLog2(val) + (val & (val - 1) ? 1 : 0);
}

/// \ingroup math
/// \brief Non-constexpr log base 2 ceiling.
inline int ceilLog2(long long val)
{
   return floorLog2(val) + (val & (val - 1) ? 1 : 0);
}
}
