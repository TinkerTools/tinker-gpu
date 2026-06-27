#pragma once

namespace tinker {
/// \ingroup math
/// \brief If `b < 0`, returns `-abs(a)`, otherwise, returns `abs(a)`.
/// Similar to the 32-bit integer version of Fortran `SIGN(A,B)`.
/// \note Standard C and CUDA libraries only have float and double versions.
__device__
inline int intCopysignShift(int a, int b)
{
   int mask = (a ^ b) >> 31;
   return (mask + a) ^ mask;
}

/// \ingroup math
/// \brief If `b < 0`, returns `-abs(a)`, otherwise, returns `abs(a)`.
__device__
inline int intCopysignIf(int a, int b)
{
   int ans = abs(a);
   if (b < 0)
      ans = -ans;
   return ans;
}

/// \def INT_COPYSIGN
/// \ingroup math
/// \brief Defines the implementation of `int copysign(int,int);` function.
#define INT_COPYSIGN intCopysignShift
}
