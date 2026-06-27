#pragma once
#include "math/sinhc.h"

namespace tinker {
#pragma ac routine seq
template <int DO_G>
SEQ_CUDA
inline void damp_hippodisp(real* restrict dmpij, real r, real rr1, real ai, real aj)
{
   real a = ai * r, b = aj * r;
   real c = (b + a) / 2, d = (b - a) / 2;
   real expmc = REAL_EXP(-c);

   real t = (ai + aj) * r;
   real x = a / t, y = 1 - x;
   real c3 = c * c * c, d2 = d * d;

   real f1d, f2d, f3d;
   if CONSTEXPR (DO_G)
      fsinhc3(d, f1d, f2d, f3d);
   else
      fsinhc2(d, f1d, f2d);

   real ec = expmc / 32, ea = REAL_EXP(-a) / 256, eb = REAL_EXP(-b) / 256;

   // [0]
   real k01, k02, l00x, l01x, l02x, l03x, l00y, l01y, l02y, l03y;
   k01 = c * (c * (c * (c + 8) + 18) + 18);
   k02 = c3 * (c * (c + 2) + 2);
   l00x = 32 * (x * (x * (2 * x - 3) - 3) + 6);
   l00y = 32 * (y * (y * (2 * y - 3) - 3) + 6);
   l01x = 4 * (8 * x * (x * (x * (2 * x - 3) - 3) + 6) - 9);
   l01y = 4 * (8 * y * (y * (y * (2 * y - 3) - 3) + 6) - 9);
   l02x = 2 * x * (8 * x * (x * (x * (2 * x - 3) - 3) + 6) - 9) - 9;
   l02y = 2 * y * (8 * y * (y * (y * (2 * y - 3) - 3) + 6) - 9) - 9;
   l03x = (-y) * (4 * (-y) * x * (4 * (-y) * x - 1) + 1);
   l03y = (-x) * (4 * (-x) * y * (4 * (-x) * y - 1) + 1);
   l01x *= t, l01y *= t;
   l02x *= t * t, l02y *= t * t;
   l03x *= t * t * t, l03y *= t * t * t;
   dmpij[0] = 1 - ((k01 * f1d + k02 * f2d) * ec + (l00x + l01x + l02x + l03x) * ea + (l00y + l01y + l02y + l03y) * eb);

   // [1]
   if CONSTEXPR (DO_G) {
      real k11, k12, k13, l10x, l11x, l12x, l13x, l10y, l11y, l12y, l13y;
      k11 = c * (c * (c * (c * (c + 4) - 6) - 18) - 18);
      k12 = c3 * (c * ((c - 3) * c - 6) - 6) - k01 * d2;
      k13 = -k02 * d2;
      l10x = a * l00x;
      l11x = (a - 1) * l01x;
      l12x = (a - 2) * l02x;
      l13x = (a - 3) * l03x;
      l10y = b * l00y;
      l11y = (b - 1) * l01y;
      l12y = (b - 2) * l02y;
      l13y = (b - 3) * l03y;
      dmpij[1] = ((k11 * f1d + k12 * f2d + k13 * f3d) * ec + (l10x + l11x + l12x + l13x) * ea + (l10y + l11y + l12y + l13y) * eb) * rr1;
   }
}
}
