#include "math/matexp.h"
#include <cmath>

namespace tinker {
template <class T>
static void matexpZero(T m[3][3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         m[i][j] = 0;
}

template <class T>
static void matexpEye(T m[3][3])
{
   matexpZero(m);
   for (int i = 0; i < 3; ++i)
      m[i][i] = 1;
}

template <class T>
static void matexpCopy(T ans[3][3], const T m[3][3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = m[i][j];
}

template <class T>
static void matexpScale(T ans[3][3], const T m[3][3], T scale)
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = scale * m[i][j];
}

template <class T>
static void matexpAdd(T ans[3][3], const T a[3][3], const T b[3][3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = a[i][j] + b[i][j];
}

template <class T>
static void matexpSub(T ans[3][3], const T a[3][3], const T b[3][3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = a[i][j] - b[i][j];
}

template <class T>
static void matexpMul(T ans[3][3], const T a[3][3], const T b[3][3])
{
   T c[3][3];
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
      }
   }
   matexpCopy(ans, c);
}

template <class T>
static void matexpAddScaled(T ans[3][3], const T a[3][3], const T b[3][3], T scale)
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = a[i][j] + scale * b[i][j];
}

template <class T>
static T matexpNorm1(const T m[3][3])
{
   T ans = 0;
   for (int j = 0; j < 3; ++j) {
      T col = std::fabs(m[0][j]) + std::fabs(m[1][j]) + std::fabs(m[2][j]);
      if (col > ans)
         ans = col;
   }
   return ans;
}

template <class T>
static void matexpSolve(T ans[3][3], const T a[3][3], const T b[3][3])
{
   T lu[3][3], rhs[3][3];
   matexpCopy(lu, a);
   matexpCopy(rhs, b);

   for (int k = 0; k < 3; ++k) {
      int ipiv = k;
      T piv = std::fabs(lu[k][k]);
      for (int i = k + 1; i < 3; ++i) {
         T val = std::fabs(lu[i][k]);
         if (val > piv) {
            piv = val;
            ipiv = i;
         }
      }

      if (ipiv != k) {
         for (int j = 0; j < 3; ++j) {
            T tmp = lu[k][j];
            lu[k][j] = lu[ipiv][j];
            lu[ipiv][j] = tmp;

            tmp = rhs[k][j];
            rhs[k][j] = rhs[ipiv][j];
            rhs[ipiv][j] = tmp;
         }
      }

      for (int i = k + 1; i < 3; ++i) {
         T mult = lu[i][k] / lu[k][k];
         lu[i][k] = 0;
         for (int j = k + 1; j < 3; ++j)
            lu[i][j] -= mult * lu[k][j];
         for (int j = 0; j < 3; ++j)
            rhs[i][j] -= mult * rhs[k][j];
      }
   }

   for (int j = 0; j < 3; ++j) {
      for (int i = 2; i >= 0; --i) {
         T sum = rhs[i][j];
         for (int k = i + 1; k < 3; ++k)
            sum -= lu[i][k] * ans[k][j];
         ans[i][j] = sum / lu[i][i];
      }
   }
}

template <class T>
void matExp3(T ans[3][3], T m[3][3], T t)
{
   // Scaling and squaring with the [13/13] Pade approximant.
   constexpr T theta13 = (T)5.371920351148152;
   constexpr T b0 = (T)1.0;
   constexpr T b1 = (T)(32382376266240000.0 / 64764752532480000.0);
   constexpr T b2 = (T)(7771770303897600.0 / 64764752532480000.0);
   constexpr T b3 = (T)(1187353796428800.0 / 64764752532480000.0);
   constexpr T b4 = (T)(129060195264000.0 / 64764752532480000.0);
   constexpr T b5 = (T)(10559470521600.0 / 64764752532480000.0);
   constexpr T b6 = (T)(670442572800.0 / 64764752532480000.0);
   constexpr T b7 = (T)(33522128640.0 / 64764752532480000.0);
   constexpr T b8 = (T)(1323241920.0 / 64764752532480000.0);
   constexpr T b9 = (T)(40840800.0 / 64764752532480000.0);
   constexpr T b10 = (T)(960960.0 / 64764752532480000.0);
   constexpr T b11 = (T)(16380.0 / 64764752532480000.0);
   constexpr T b12 = (T)(182.0 / 64764752532480000.0);
   constexpr T b13 = (T)(1.0 / 64764752532480000.0);

   T a[3][3];
   matexpScale(a, m, t);

   int scale = 0;
   T norm = matexpNorm1(a);
   if (std::isfinite(norm) and norm > theta13)
      scale = (int)std::ceil(std::log2(norm / theta13));
   if (scale > 0) {
      T s = std::ldexp((T)1, -scale);
      matexpScale(a, a, s);
   }

   T ident[3][3], a2[3][3], a4[3][3], a6[3][3];
   matexpEye(ident);
   matexpMul(a2, a, a);
   matexpMul(a4, a2, a2);
   matexpMul(a6, a4, a2);

   T tmp0[3][3], tmp1[3][3], tmp2[3][3], u[3][3], v[3][3];

   matexpAddScaled(tmp0, a6, a4, b11 / b13);
   matexpAddScaled(tmp0, tmp0, a2, b9 / b13);
   matexpScale(tmp0, tmp0, b13);
   matexpMul(tmp1, a6, tmp0);
   matexpAddScaled(tmp1, tmp1, a6, b7);
   matexpAddScaled(tmp1, tmp1, a4, b5);
   matexpAddScaled(tmp1, tmp1, a2, b3);
   matexpAddScaled(tmp1, tmp1, ident, b1);
   matexpMul(u, a, tmp1);

   matexpAddScaled(tmp0, a6, a4, b10 / b12);
   matexpAddScaled(tmp0, tmp0, a2, b8 / b12);
   matexpScale(tmp0, tmp0, b12);
   matexpMul(tmp1, a6, tmp0);
   matexpAddScaled(tmp1, tmp1, a6, b6);
   matexpAddScaled(tmp1, tmp1, a4, b4);
   matexpAddScaled(tmp1, tmp1, a2, b2);
   matexpAddScaled(v, tmp1, ident, b0);

   T p[3][3], q[3][3];
   matexpAdd(p, v, u);
   matexpSub(q, v, u);
   matexpSolve(ans, q, p);

   for (int i = 0; i < scale; ++i) {
      matexpCopy(tmp2, ans);
      matexpMul(ans, tmp2, tmp2);
   }
}
template void matExp3(double ans[3][3], double m[3][3], double t);

template <>
void matExp3(float ans[3][3], float m[3][3], float t)
{
   double md[3][3], ansd[3][3];
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         md[i][j] = m[i][j];

   matExp3(ansd, md, (double)t);

   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ans[i][j] = (float)ansd[i][j];
}
}
