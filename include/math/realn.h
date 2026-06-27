#pragma once
#include "ff/precision.h"
#include "realndef.h"
#include "seq/seq.h"

namespace tinker {
/// \ingroup math
/// \{
#if TINKER_REAL_SIZE == 8
using real2 = double2;
using real3 = double3;
using real4 = double4;
#define make_real2(x, y)       make_double2((x), (y))
#define make_real3(x, y, z)    make_double3((x), (y), (z))
#define make_real4(x, y, z, w) make_double4((x), (y), (z), (w))
#elif TINKER_REAL_SIZE == 4
using real2 = float2;
using real3 = float3;
using real4 = float4;
#define make_real2(x, y)       make_float2((x), (y))
#define make_real3(x, y, z)    make_float3((x), (y), (z))
#define make_real4(x, y, z, w) make_float4((x), (y), (z), (w))
#endif

// -
SEQ_ROUTINE
inline real3 operator-(real3 a)
{
   return make_real3(-a.x, -a.y, -a.z);
}

// + - *
SEQ_ROUTINE
inline real3 operator+(real3 a, real3 b)
{
   return make_real3(a.x + b.x, a.y + b.y, a.z + b.z);
}

SEQ_ROUTINE
inline real3 operator-(real3 a, real3 b)
{
   return make_real3(a.x - b.x, a.y - b.y, a.z - b.z);
}

SEQ_ROUTINE
inline real3 operator*(real a, real3 b)
{
   return make_real3(a * b.x, a * b.y, a * b.z);
}

SEQ_ROUTINE
inline real3 operator*(real3 b, real a)
{
   return make_real3(a * b.x, a * b.y, a * b.z);
}

// += -= *=
SEQ_ROUTINE
inline void operator+=(real3& a, real3 b)
{
   a.x += b.x;
   a.y += b.y;
   a.z += b.z;
}

SEQ_ROUTINE
inline void operator-=(real3& a, real3 b)
{
   a.x -= b.x;
   a.y -= b.y;
   a.z -= b.z;
}

SEQ_ROUTINE
inline void operator*=(real3& a, real b)
{
   a.x *= b;
   a.y *= b;
   a.z *= b;
}

// dot
SEQ_ROUTINE
inline real dot3(real3 a, real3 b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

SEQ_ROUTINE
inline real dot3(real3 a, real bx, real by, real bz)
{
   return a.x * bx + a.y * by + a.z * bz;
}

SEQ_ROUTINE
inline real dot3(real ax, real ay, real az, real3 b)
{
   return ax * b.x + ay * b.y + az * b.z;
}

SEQ_ROUTINE
inline real dot3(real ax, real ay, real az, real bx, real by, real bz)
{
   return ax * bx + ay * by + az * bz;
}

// symmetric matrix(3,3) dot vector(3)
SEQ_ROUTINE
inline real3 matvec(real xx, real xy, real xz, real yy, real yz, real zz, real3 v)
{
   return make_real3(dot3(xx, xy, xz, v), dot3(xy, yy, yz, v), dot3(xz, yz, zz, v));
}

// cross product
SEQ_ROUTINE
inline real3 cross(real ax, real ay, real az, real bx, real by, real bz)
{
   return make_real3(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
}

SEQ_ROUTINE
inline real3 cross(real ax, real ay, real az, real3 b)
{
   return cross(ax, ay, az, b.x, b.y, b.z);
}

SEQ_ROUTINE
inline real3 cross(real3 a, real bx, real by, real bz)
{
   return cross(a.x, a.y, a.z, bx, by, bz);
}

SEQ_ROUTINE
inline real3 cross(real3 a, real3 b)
{
   return cross(a.x, a.y, a.z, b.x, b.y, b.z);
}
/// \}
}
