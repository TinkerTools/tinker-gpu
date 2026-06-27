#pragma once
#include "math/sinhc.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <int DirOrder, int MutOrder>
SEQ_CUDA
inline void damp_gordon1(real* restrict dmpij, real* restrict dmpi, real* restrict dmpj, real r, real ai, real aj)
{
   const real div3 = 1 / ((real)3);
   const real div6 = 1 / ((real)6);
   const real div15 = 1 / ((real)15);
   const real div30 = 1 / ((real)30);
   const real div105 = 1 / ((real)105);
   const real div945 = 1 / ((real)945);

   real a, expi, b, expk;
   a = ai * r, b = aj * r;
   expi = REAL_EXP(-a), expk = REAL_EXP(-b);

   // dmpi, dmpj
   real a2, a3, a4, a5, b2, b3, b4, b5;

   if CONSTEXPR (DirOrder >= 1) {
      dmpi[0] = 1 - (1 + 0.5f * a) * expi;
      dmpj[0] = 1 - (1 + 0.5f * b) * expk;
   }
   if CONSTEXPR (DirOrder >= 3) {
      a2 = a * a, b2 = b * b;
      dmpi[1] = 1 - (1 + a + 0.5f * a2) * expi;
      dmpj[1] = 1 - (1 + b + 0.5f * b2) * expk;
   }
   if CONSTEXPR (DirOrder >= 5) {
      a3 = a * a2, b3 = b * b2;
      dmpi[2] = 1 - (1 + a + 0.5f * a2 + a3 * div6) * expi;
      dmpj[2] = 1 - (1 + b + 0.5f * b2 + b3 * div6) * expk;
   }
   if CONSTEXPR (DirOrder >= 7) {
      a4 = a2 * a2, b4 = b2 * b2;
      dmpi[3] = 1 - (1 + a + 0.5f * a2 + a3 * div6 + a4 * div30) * expi;
      dmpj[3] = 1 - (1 + b + 0.5f * b2 + b3 * div6 + b4 * div30) * expk;
   }
   if CONSTEXPR (DirOrder >= 9) {
      a5 = a2 * a3, b5 = b2 * b3;
      dmpi[4] = 1 - (1 + a + 0.5f * a2 + a3 * div6 + (4 * a4 + 0.5f * a5) * div105) * expi;
      dmpj[4] = 1 - (1 + b + 0.5f * b2 + b3 * div6 + (4 * b4 + 0.5f * b5) * div105) * expk;
   }

   // dmpij
   if CONSTEXPR (MutOrder >= 1) {
      real c = (b + a) / 2, d = (b - a) / 2;
      real expmc = REAL_EXP(-c);
      real t = (ai + aj) * r;
      real x = a / t, y = 1 - x;
      real ec = expmc / 16, ea = expi / 32, eb = expk / 32;

      real c2 = c * c, c3 = c * c * c;
      real d2 = d * d;
      real c2d2 = c2 * d2;

      real f1d, f2d, f3d, f4d, f5d, f6d, f7d;
      if CONSTEXPR (MutOrder >= 11)
         fsinhc7(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
      else if CONSTEXPR (MutOrder >= 9)
         fsinhc6(d, f1d, f2d, f3d, f4d, f5d, f6d);
      else if CONSTEXPR (MutOrder >= 7)
         fsinhc5(d, f1d, f2d, f3d, f4d, f5d);
      else if CONSTEXPR (MutOrder >= 5)
         fsinhc4(d, f1d, f2d, f3d, f4d);
      else if CONSTEXPR (MutOrder >= 3)
         fsinhc3(d, f1d, f2d, f3d);
      else
         fsinhc2(d, f1d, f2d);

#define TINKER_GORDON1_L00(X) (4 * ((X) * ((X) * (2 * (X)-3) - 3) + 6))
#define TINKER_GORDON1_L01(X) ((X) * (4 * ((X)-3) * (X) + 11) - 2)

#define TINKER_GORDON1_M0(a) (1)
#define TINKER_GORDON1_M1(a) ((a) + 1)
#define TINKER_GORDON1_M2(a) ((a) * ((a) + 3) + 3)
#define TINKER_GORDON1_M3(a) ((a) * ((a) * ((a) + 6) + 15) + 15)
#define TINKER_GORDON1_M4(a) ((a) * ((a) * ((a) * ((a) + 10) + 45) + 105) + 105)
#define TINKER_GORDON1_M5(a) ((a) * ((a) * ((a) * ((a) * ((a) + 15) + 105) + 420) + 945) + 945)

      real l00x, l01x, l00y, l01y;
      if CONSTEXPR (MutOrder >= 1) {
         real k01, k02;
         k01 = 3 * c * (c + 3);
         k02 = c3;
         l00x = TINKER_GORDON1_L00(x), l01x = TINKER_GORDON1_L01(x) * t;
         l00y = TINKER_GORDON1_L00(y), l01y = TINKER_GORDON1_L01(y) * t;
         dmpij[0] = 1 - ((k01 * f1d + k02 * f2d) * ec + (l00x + l01x) * ea + (l00y + l01y) * eb);
      }
      if CONSTEXPR (MutOrder >= 3) {
         real k11, k12, k13, l10x, l11x, l10y, l11y;
         k11 = 3 * c2 * (c + 2);
         k12 = c * ((c - 2) * c2 - 3 * (c + 3) * d2);
         k13 = -c3 * d2;
         l10x = TINKER_GORDON1_M1(a) * l00x, l11x = a * TINKER_GORDON1_M0(a) * l01x;
         l10y = TINKER_GORDON1_M1(b) * l00y, l11y = b * TINKER_GORDON1_M0(b) * l01y;
         dmpij[1] = 1 - ((k11 * f1d + k12 * f2d + k13 * f3d) * ec + (l10x + l11x) * ea + (l10y + l11y) * eb);
      }
      if CONSTEXPR (MutOrder >= 5) {
         real k21, k22, k23, k24, l20x, l21x, l20y, l21y;
         k21 = 3 * c2 * (c * (c + 2) + 2);
         k22 = c2 * ((c - 3) * c2 - 6 * (c + 2) * d2);
         k23 = 3 * (c + 3) * d2 - 2 * (c - 2) * c2;
         k24 = c2d2;
         l20x = TINKER_GORDON1_M2(a) * l00x, l21x = a * TINKER_GORDON1_M1(a) * l01x;
         l20y = TINKER_GORDON1_M2(b) * l00y, l21y = b * TINKER_GORDON1_M1(b) * l01y;
         dmpij[2] = 1 - div3 * ((k21 * f1d + k22 * f2d + c * d2 * (k23 * f3d + k24 * f4d)) * ec + (l20x + l21x) * ea + (l20y + l21y) * eb);
      }
      if CONSTEXPR (MutOrder >= 7) {
         real k31, k32, k33, k34, k35, l30x, l31x, l30y, l31y;
         k31 = 3 * c2 * (c * (c * (c + 3) + 6) + 6);
         k32 = c2 * (c2 * ((c - 3) * c - 3) - 9 * (c * (c + 2) + 2) * d2);
         k33 = c2 * (9 * (c + 2) * d2 - 3 * (c - 3) * c2);
         k34 = 3 * (c - 2) * c3 - 3 * c * (c + 3) * d2;
         k35 = -c3 * d2;
         l30x = TINKER_GORDON1_M3(a) * l00x, l31x = a * TINKER_GORDON1_M2(a) * l01x;
         l30y = TINKER_GORDON1_M3(b) * l00y, l31y = b * TINKER_GORDON1_M2(b) * l01y;
         dmpij[3] = 1 - div15 * ((k31 * f1d + k32 * f2d + d2 * (k33 * f3d + d2 * (k34 * f4d + k35 * f5d))) * ec + (l30x + l31x) * ea + (l30y + l31y) * eb);
      }
      if CONSTEXPR (MutOrder >= 9) {
         real k41, k42, k43, k44, k45, k46, l40x, l41x, l40y, l41y;
         k41 = 3 * c2 * (c * (c * (c * (c + 5) + 15) + 30) + 30);
         k42 = c2 * (c2 * (c * ((c - 2) * c - 9) - 9) - 12 * (c * (c * (c + 3) + 6) + 6) * d2);
         k43 = c2 * (18 * (c * (c + 2) + 2) * d2 - 4 * c2 * ((c - 3) * c - 3));
         k44 = c2 * (6 * (c - 3) * c2 - 12 * (c + 2) * d2);
         k45 = c * (3 * (c + 3) * d2 - 4 * (c - 2) * c2);
         k46 = c3 * d2;
         l40x = TINKER_GORDON1_M4(a) * l00x, l41x = a * TINKER_GORDON1_M3(a) * l01x;
         l40y = TINKER_GORDON1_M4(b) * l00y, l41y = b * TINKER_GORDON1_M3(b) * l01y;
         dmpij[4] = 1
            - div105
               * ((k41 * f1d + k42 * f2d + d2 * (k43 * f3d + d2 * (k44 * f4d + d2 * (k45 * f5d + k46 * f6d)))) * ec + (l40x + l41x) * ea + (l40y + l41y) * eb);
      }
      if CONSTEXPR (MutOrder >= 11) {
         real k51, k52, k53, k54, k55, k56, k57, l50x, l51x, l50y, l51y;
         k51 = 3 * c2 * (c * (c * (c * (c * (c + 8) + 35) + 105) + 210) + 210);
         k52 = c2 * (c2 * (c * (c3 - 15 * c - 45) - 45) - 15 * (c * (c * (c * (c + 5) + 15) + 30) + 30) * d2);
         k53 = c2 * (5 * (c * (9 - (c - 2) * c) + 9) * c2 + 30 * (c * (c * (c + 3) + 6) + 6) * d2);
         k54 = c2 * (10 * c2 * ((c - 3) * c - 3) - 30 * (c * (c + 2) + 2) * d2);
         k55 = c2 * (15 * (c + 2) * d2 - 10 * (c - 3) * c2);
         k56 = c * (5 * (c - 2) * c2 - 3 * (c + 3) * d2);
         k57 = -c3 * d2;
         l50x = TINKER_GORDON1_M5(a) * l00x, l51x = a * TINKER_GORDON1_M4(a) * l01x;
         l50y = TINKER_GORDON1_M5(b) * l00y, l51y = b * TINKER_GORDON1_M4(b) * l01y;
         dmpij[5] = 1
            - div945
               * ((k51 * f1d + k52 * f2d + d2 * (k53 * f3d + d2 * (k54 * f4d + d2 * (k55 * f5d + d2 * (k56 * f6d + k57 * f7d))))) * ec + (l50x + l51x) * ea
                  + (l50y + l51y) * eb);
      }

#undef TINKER_GORDON1_L00
#undef TINKER_GORDON1_L01
#undef TINKER_GORDON1_M0
#undef TINKER_GORDON1_M1
#undef TINKER_GORDON1_M2
#undef TINKER_GORDON1_M3
#undef TINKER_GORDON1_M4
#undef TINKER_GORDON1_M5
   }
}

/// deprecated
#pragma acc routine seq
template <int ORDER>
SEQ_CUDA
inline void damp_pole_deprecated(real* restrict dmpik, real* restrict dmpi, real* restrict dmpk, real r, real alphai, real alphak)
{
#if TINKER_REAL_SIZE == 8
   real eps = 0.001f;
#elif TINKER_REAL_SIZE == 4
   real eps = 0.05f;
#endif

   real diff = REAL_ABS(alphai - alphak);

   if (diff < eps)
      alphai = 0.5f * (alphai + alphak);

   real dampi = alphai * r;
   real dampk = alphak * r;
   real expi = REAL_EXP(-dampi);
   real expk = REAL_EXP(-dampk);

   real dampi2 = dampi * dampi;
   real dampi3 = dampi * dampi2;
   real dampi4 = dampi2 * dampi2;
   real dampi5 = dampi2 * dampi3;

   // divisions
   const real div3 = 1 / ((real)3);
   const real div6 = 1 / ((real)6);
   const real div15 = 1 / ((real)15);
   const real div21 = 1 / ((real)21);
   const real div30 = 1 / ((real)30);
   const real div105 = 1 / ((real)105);

   // GORDON1
   // core-valence
   dmpi[0] = (1 + 0.5f * dampi) * expi;
   dmpi[1] = (1 + dampi + 0.5f * dampi2) * expi;
   dmpi[2] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6) * expi;
   dmpi[3] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div30) * expi;
   dmpi[4] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + (4 * dampi4 + 0.5f * dampi5) * div105) * expi;

   if (diff < eps) {
      dmpk[0] = dmpi[0];
      dmpk[1] = dmpi[1];
      dmpk[2] = dmpi[2];
      dmpk[3] = dmpi[3];
      dmpk[4] = dmpi[4];

      // valence-valence
      real dampi6 = dampi3 * dampi3;
      real dampi7 = dampi3 * dampi4;
      const real div5 = 1 / ((real)5);
      const real div16 = 1 / ((real)16);
      const real div24 = 1 / ((real)24);
      const real div42 = 1 / ((real)42);
      const real div48 = 1 / ((real)48);
      const real div120 = 1 / ((real)120);
      const real div144 = 1 / ((real)144);

      dmpik[0] = (1 + (11 * dampi + 3 * dampi2) * div16 + dampi3 * div48) * expi;
      dmpik[1] = (1 + dampi + 0.5f * dampi2 + (7 * dampi3 + dampi4) * div48) * expi;
      dmpik[2] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div24 + dampi5 * div144) * expi;
      dmpik[3] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div24 + (dampi5 + dampi6 * div6) * div120) * expi;
      dmpik[4] = ((1 + dampi + 0.5f * dampi2 + dampi3 * div6) + ((dampi4 + dampi5 * div5) + 0.1f * (dampi6 * div3 + dampi7 * div21)) * div24) * expi;
      if CONSTEXPR (ORDER > 9) {
         real dampi8 = dampi4 * dampi4;
         const real div378 = 1 / ((real)378);
         dmpik[5] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div24 + (dampi5 + dampi6 * div6 + dampi7 * div42 + dampi8 * div378) * div120) * expi;
      }
   } else {
      real dampk2 = dampk * dampk;
      real dampk3 = dampk * dampk2;
      real dampk4 = dampk2 * dampk2;
      real dampk5 = dampk2 * dampk3;

      const real div5 = 1 / ((real)5);
      const real div7 = 1 / ((real)7);

      dmpk[0] = (1 + 0.5f * dampk) * expk;
      dmpk[1] = (1 + dampk + 0.5f * dampk2) * expk;
      dmpk[2] = (1 + dampk + 0.5f * dampk2 + dampk3 * div6) * expk;
      dmpk[3] = (1 + dampk + 0.5f * dampk2 + dampk3 * div6 + dampk4 * div30) * expk;
      dmpk[4] = (1 + dampk + 0.5f * dampk2 + dampk3 * div6 + (4 * dampk4 + 0.5f * dampk5) * div105) * expk;

      // valence-valence
      real alphai2 = alphai * alphai;
      real alphak2 = alphak * alphak;
      real alphaik = ((alphak + alphai) * (alphak - alphai));
      real termi = alphak2 / alphaik;
      real termk = -alphai2 / alphaik;
      real termi2 = termi * termi;
      real termk2 = termk * termk;

      dmpik[0] = termi2 * (1 + 2 * termk + 0.5f * dampi) * expi + termk2 * (1 + 2 * termi + 0.5f * dampk) * expk;
      dmpik[1] = termi2 * (1 + dampi + 0.5f * dampi2) * expi + termk2 * (1 + dampk + 0.5f * dampk2) * expk + 2 * termi2 * termk * (1 + dampi) * expi
         + 2 * termk2 * termi * (1 + dampk) * expk;
      dmpik[2] = termi2 * (1 + dampi + 0.5f * dampi2 + dampi3 * div6) * expi + termk2 * (1 + dampk + 0.5f * dampk2 + dampk3 * div6) * expk
         + 2 * termi2 * termk * (1 + dampi + dampi2 * div3) * expi + 2 * termk2 * termi * (1 + dampk + dampk2 * div3) * expk;
      dmpik[3] = termi2 * (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div30) * expi
         + termk2 * (1 + dampk + 0.5f * dampk2 + dampk3 * div6 + dampk4 * div30) * expk
         + 2 * termi2 * termk * (1 + dampi + 2 * dampi2 * div5 + dampi3 * div15) * expi
         + 2 * termk2 * termi * (1 + dampk + 2 * dampk2 * div5 + dampk3 * div15) * expk;
      dmpik[4] = termi2 * (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + (4 * dampi4 + 0.5f * dampi5) * div105) * expi
         + termk2 * (1 + dampk + 0.5f * dampk2 + dampk3 * div6 + (4 * dampk4 + 0.5f * dampk5) * div105) * expk
         + 2 * termi2 * termk * (1 + dampi + 3 * dampi2 * div7 + 2 * dampi3 * div21 + dampi4 * div105) * expi
         + 2 * termk2 * termi * (1 + dampk + 3 * dampk2 * div7 + 2 * dampk3 * div21 + dampk4 * div105) * expk;

      if CONSTEXPR (ORDER > 9) {
         real dampi6 = dampi3 * dampi3;
         real dampk6 = dampk3 * dampk3;
         const real div945 = 1 / ((real)945);
         const real div9 = 1 / ((real)9);
         const real div63 = 1 / ((real)63);
         const real div126 = 1 / ((real)126);
         const real div315 = 1 / ((real)315);
         const real div1890 = 1 / ((real)1890);

         dmpik[5] = (1 + dampi + 0.5f * dampi2 + dampi3 * div6 + 5 * dampi4 * div126 + 2 * dampi5 * div315 + dampi6 * div1890) * termi2 * expi
            + (1 + dampk + 0.5f * dampk2 + dampk3 * div6 + 5 * dampk4 * div126 + 2 * dampk5 * div315 + dampk6 * div1890) * termk2 * expk
            + (1 + dampi + (4 * dampi2 + dampi3) * div9 + dampi4 * div63 + dampi5 * div945) * 2 * termi2 * termk * expi
            + (1 + dampk + (4 * dampk2 + dampk3) * div9 + dampk4 * div63 + dampk5 * div945) * 2 * termk2 * termi * expk;
      }
   }
}

#pragma acc routine seq
template <int order>
SEQ_CUDA
inline void damp_pole_v2(real* restrict dmpij, real* restrict dmpi, real* restrict dmpj, real r, real ai, real aj)
{
   damp_gordon1<9, order>(dmpij, dmpi, dmpj, r, ai, aj);
}

SEQ_ROUTINE
inline void damp_dir(real* restrict dmpi, real* restrict dmpk, real r, real alphai, real alphak)
{
   damp_gordon1<7, 0>(nullptr, dmpi, dmpk, r, alphai, alphak);
}

SEQ_ROUTINE
inline void damp_mut(real* restrict dmpik, real r, real alphai, real alphak)
{
   damp_gordon1<0, 5>(dmpik, nullptr, nullptr, r, alphai, alphak);
}

/**
 * deprecated
 */
#pragma acc routine seq
template <int order>
SEQ_CUDA
inline void damp_rep_deprecated(real* restrict dmpik, real r, real rinv, real r2, real rr3, real rr5, real rr7, real rr9, real rr11, real dmpi, real dmpk)
{
#if TINKER_REAL_SIZE == 8
   real eps = 0.001f;
#elif TINKER_REAL_SIZE == 4
   real eps = 0.05f;
#endif

   // This makes sure that dmpi > dmpk, which solves numerical issues
   // with atomi,atomk vs. atomk,atomi.
   real dmpsmall = REAL_MIN(dmpi, dmpk);
   real dmpbig = REAL_MAX(dmpi, dmpk);
   dmpi = dmpsmall;
   dmpk = dmpbig;

   real diff = REAL_ABS(dmpi - dmpk);
   if (diff < eps)
      dmpi = 0.5f * (dmpi + dmpk);

   real r3 = r2 * r;
   real r4 = r2 * r2;
   real r5 = r3 * r2;
   real dmpi2 = 0.5f * dmpi;
   real dampi = dmpi2 * r;
   real expi = REAL_EXP(-dampi);
   real dmpi22 = dmpi2 * dmpi2;
   real dmpi23 = dmpi22 * dmpi2;
   real dmpi24 = dmpi22 * dmpi22;
   real dmpi25 = dmpi23 * dmpi22;

   // divisions
   const real div3 = 1 / ((real)3);
   const real div9 = 1 / ((real)9);
   const real div945 = 1 / ((real)945);

   real pre, s, ds, d2s, d3s, d4s, d5s;
   if (diff < eps) {
      real r6 = r3 * r3;
      real r7 = r4 * r3;

      real dmpi26 = dmpi23 * dmpi23;

      pre = 2;
      s = (r + dmpi2 * r2 + dmpi22 * r3 * div3) * expi;
      ds = (dmpi22 * r3 + dmpi23 * r4) * expi * div3;
      d2s = dmpi24 * expi * r5 * div9;
      d3s = dmpi25 * expi * r6 / ((real)45);
      d4s = (dmpi25 * r6 + dmpi26 * r7) * expi / ((real)315);

      if CONSTEXPR (order > 9) {
         real r8 = r4 * r4;
         real dmpi27 = dmpi24 * dmpi23;
         d5s = (dmpi25 * r6 + dmpi26 * r7 + dmpi27 * r8 * div3) * expi * div945;
      }
   } else {
      // treat the case where alpha damping exponents are unequal

      // divisions
      real div5 = 1 / ((real)5);
      real div7 = 1 / ((real)7);
      real div15 = 1 / ((real)15);
      real div21 = 1 / ((real)21);
      real div63 = 1 / ((real)63);
      real div105 = 1 / ((real)105);
      real div189 = 1 / ((real)189);

      real dmpk2 = 0.5f * dmpk;
      real dampk = dmpk2 * r;
      real expk = REAL_EXP(-dampk);
      real dmpk22 = dmpk2 * dmpk2;
      real dmpk23 = dmpk22 * dmpk2;
      real dmpk24 = dmpk22 * dmpk22;
      real dmpk25 = dmpk23 * dmpk22;

      real term = 0.25f * (dmpi + dmpk) * (dmpi - dmpk);
      real term1 = (dmpi + dmpk) * (dmpi - dmpk);
      real tmp = 4 * (dmpi * dmpk) / term1;
      pre = (128 * dmpi23 * dmpk23) / (term * term * term * term);

      real coef1 = 16 / term1;

      s = (dampi * expk) + (dampk * expi) + tmp * (expi - expk);
      ds = (term * dmpk2 * r2 - 4 * (dmpk22 * r + dmpk2)) * dmpi2 * expk / term + (term * dmpi2 * r2 + 4 * (dmpi22 * r + dmpi2)) * dmpk2 * expi / term;

      d2s = ((dmpk2 * r2 + dmpk22 * r3) * div3 - coef1 * (dmpk23 * r2 * div3 + dmpk22 * r + dmpk2)) * dmpi2 * expk
         + ((dmpi2 * r2 + dmpi22 * r3) * div3 + coef1 * (dmpi23 * r2 * div3 + dmpi22 * r + dmpi2)) * dmpk2 * expi;

      d3s = ((dmpk23 * r4 * div3 + dmpk22 * r3 + dmpk2 * r2) * div5 - coef1 * (dmpk24 * r3 * div15 + (2 * div5) * dmpk23 * r2 + dmpk22 * r + dmpk2)) * dmpi2
            * expk
         + ((dmpi23 * r4 * div3 + dmpi22 * r3 + dmpi2 * r2) * div5 + coef1 * (dmpi24 * r3 * div15 + (2 * div5) * dmpi23 * r2 + dmpi22 * r + dmpi2)) * dmpk2
            * expi;

      d4s = ((dmpk24 * r5 * div15 + 2 * div5 * dmpk23 * r4 + dmpk22 * r3 + dmpk2 * r2) * div7
             - coef1 * (dmpk25 * r4 * div105 + 2 * div21 * dmpk24 * r3 + 3 * div7 * dmpk23 * r2 + dmpk22 * r + dmpk2))
            * dmpi2 * expk
         + ((dmpi24 * r5 * div15 + 2 * div5 * dmpi23 * r4 + dmpi22 * r3 + dmpi2 * r2) * div7
            + coef1 * (dmpi25 * r4 * div105 + 2 * div21 * dmpi24 * r3 + 3 * div7 * dmpi23 * r2 + dmpi22 * r + dmpi2))
            * dmpk2 * expi;

      if CONSTEXPR (order > 9) {
         real r6 = r3 * r3;
         real dmpi26 = dmpi23 * dmpi23;
         real dmpk26 = dmpk23 * dmpk23;
         d5s = (dmpk25 * r6 * div945 + 2 * div189 * dmpk24 * r5 + dmpk23 * r4 * div21 + dmpk22 * r3 * div9 + dmpk2 * r2 * div9
                - coef1 * (dmpk26 * r5 * div945 + dmpk25 * r4 * div63 + dmpk24 * r3 * div9 + 4 * div9 * dmpk23 * r2 + dmpk22 * r + dmpk2))
               * dmpi2 * expk
            + (dmpi25 * r6 * div945 + 2 * div189 * dmpi24 * r5 + dmpi23 * r4 * div21 + dmpi22 * r3 * div9 + dmpi2 * r2 * div9
               + coef1 * (dmpi26 * r5 * div945 + dmpi25 * r4 * div63 + dmpi24 * r3 * div9 + 4 * div9 * dmpi23 * r2 + dmpi22 * r + dmpi2))
               * dmpk2 * expi;
      }
   }

   // convert partial derivatives into full derivatives
   s *= rinv;
   ds *= rr3;
   d2s *= rr5;
   d3s *= rr7;
   d4s *= rr9;

   dmpik[0] = 0.5f * pre * s * s;
   dmpik[1] = pre * s * ds;
   dmpik[2] = pre * (s * d2s + ds * ds);
   dmpik[3] = pre * (s * d3s + 3 * ds * d2s);
   dmpik[4] = pre * (s * d4s + 4 * ds * d3s + 3 * d2s * d2s);

   if CONSTEXPR (order > 9) {
      d5s *= rr11;
      dmpik[5] = pre * (s * d5s + 5 * ds * d4s + 10 * d2s * d3s);
   }
}

/**
 * rr1: 1/r
 * ai: dmpi
 * aj: dmpk
 */
#pragma acc routine seq
template <int order>
SEQ_CUDA
inline void damp_rep(real* restrict dmpik, real r, real rr1, real r2, real rr3, real rr5, real rr7, real rr9, real rr11, real ai /* dmpi */, real aj /* dmpk */)
{
   real pfac = 2 / (ai + aj);
   pfac = pfac * pfac;
   pfac = pfac * ai * aj;
   pfac = pfac * pfac * pfac;
   pfac *= r2;

   real a = ai * r / 2, b = aj * r / 2;
   real c = (a + b) / 2, d = (b - a) / 2;
   real expmc = REAL_EXP(-c);

   real c2 = c * c;
   real c3 = c2 * c;
   real c4 = c2 * c2;
   real d2 = d * d;
   real d4 = d2 * d2;
   real c2d2 = (c * d) * (c * d);

   real f1d, f2d, f3d, f4d, f5d, f6d, f7d;
   if CONSTEXPR (order > 9)
      fsinhc7(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
   else
      fsinhc6(d, f1d, f2d, f3d, f4d, f5d, f6d);

   real inv3 = 1. / 3, inv15 = 1. / 15, inv105 = 1. / 105, inv945 = 1. / 945;

   // compute
   real s;
   s = f1d * (c + 1) + f2d * c2;
   s *= rr1;
   s *= expmc;
   dmpik[0] = pfac * s * s;

   real ds;
   ds = f1d * c2 + f2d * ((c - 2) * c2 - (c + 1) * d2) - f3d * c2d2;
   ds *= rr3;
   ds *= expmc;
   dmpik[1] = pfac * 2 * s * ds;

   real d2s = 0;
   d2s += f1d * c3 + f2d * c2 * ((c - 3) * c - 2 * d2);
   d2s += d2 * (f3d * (2 * (2 - c) * c2 + (c + 1) * d2) + f4d * c2d2);
   d2s *= rr5 * inv3;
   d2s *= expmc;
   dmpik[2] = pfac * 2 * (s * d2s + ds * ds);

   real d3s = 0;
   d3s += f1d * c3 * (c + 1) + f2d * c3 * (c * (c - 3) - 3 * (d2 + 1));
   d3s -= d2 * (f3d * 3 * c2 * ((c - 3) * c - d2) + d2 * (f4d * (3 * (2 - c) * c2 + (c + 1) * d2) + f5d * c2d2));
   d3s *= rr7 * inv15;
   d3s *= expmc;
   dmpik[3] = pfac * 2 * (s * d3s + 3 * ds * d2s);

   real d4s = 0;
   d4s += f1d * c3 * (3 + c * (c + 3)) + f2d * c3 * (c3 - 9 * (c + 1) - 2 * c2 - 4 * (c + 1) * d2);
   d4s += d2
      * (f3d * 2 * c3 * (6 * (c + 1) - 2 * c2 + 3 * d2)
         + d2 * (f4d * 2 * c2 * (3 * (c - 3) * c - 2 * d2) + f5d * d2 * (4 * (2 - c) * c2 + (c + 1) * d2) + f6d * c2 * d4));
   d4s *= rr9 * inv105;
   d4s *= expmc;
   dmpik[4] = pfac * 2 * (s * d4s + 4 * ds * d3s + 3 * d2s * d2s);

   if CONSTEXPR (order > 9) {
      real d5s = 0;
      d5s += f1d * c3 * (15 + c * (15 + c * (c + 6)));
      d5s += f2d * c3 * (c4 - 15 * c2 - 45 * (c + 1) - 5 * (3 + c * (c + 3)) * d2);
      d5s -= d2
         * (f3d * 5 * c3 * (c3 - 9 * (c + 1) - 2 * c2 - 2 * (c + 1) * d2)
            + d2
               * (f4d * 10 * c3 * (3 - (c - 3) * c + d2)
                  + d2 * (f5d * 5 * c2 * (2 * (c - 3) * c - d2) + f6d * d2 * ((c + 1) * d2 - 5 * (c - 2) * c2) + f7d * c2 * d4)));
      d5s *= rr11 * inv945;
      d5s *= expmc;
      dmpik[5] = pfac * 2 * (s * d5s + 5 * ds * d4s + 10 * d2s * d3s);
   }
}
}
